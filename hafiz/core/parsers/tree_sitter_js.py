"""JavaScript / TypeScript parser — tree-sitter based.

Mirrors `python_ast.py`'s contract for the JS/TS family. Emits these
unit kinds:
    code.module     one per file (always, even when parsing degrades)
    code.class      class declarations (incl. abstract/exported)
    code.function   top-level functions and top-level arrow/function
                    expressions bound to a name (`const f = () => {}`)
    code.method     methods defined inside a class body

And these edge relations:
    calls           caller function/method/module -> callee
    imports         module -> imported specifier (ES `import` + `require`)
    inherits        class -> base class (`extends`)

Qualified names follow the same convention as the Python parser:
`OuterClass.method`, `Outer.Inner`. Line numbers are 1-indexed, inclusive.

tree-sitter is an OPTIONAL dependency (`pip install hafiz[js]`). When it
is not importable, this module's `AVAILABLE` flag is False and
`_build_registry` skips registration, so `.js`/`.ts` files fall through
to `WholeFileParser` — no crash, no hard dependency. tree-sitter itself
never raises on malformed input (it produces ERROR nodes); we still wrap
parsing so any unexpected failure degrades to a module-only unit.

Covered extensions:
    .js .jsx .mjs .cjs   (JavaScript grammar; JSX handled by the JS grammar)
    .ts .mts .cts        (TypeScript grammar)
    .tsx                 (TSX grammar — TS + JSX)
"""

from __future__ import annotations

import logging
from pathlib import Path

from hafiz.core.parsers import ParsedEdge, ParsedUnit, ParseResult

logger = logging.getLogger("hafiz.parsers")

# ---------------------------------------------------------------------------
# Optional dependency wiring — import tree-sitter + grammars, or stay dark.
# ---------------------------------------------------------------------------

try:
    import tree_sitter_javascript as _tsjs
    import tree_sitter_typescript as _tsts
    from tree_sitter import Language as _Language
    from tree_sitter import Parser as _TSParser

    _JS_LANGUAGE = _Language(_tsjs.language())
    _TS_LANGUAGE = _Language(_tsts.language_typescript())
    _TSX_LANGUAGE = _Language(_tsts.language_tsx())
    AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised via import-failure path
    AVAILABLE = False
    logger.debug("tree-sitter JS/TS grammars unavailable; parser will not register: %s", exc)


_TS_EXTENSIONS = {".ts", ".tsx", ".mts", ".cts"}


def _language_for(ext: str):
    """Pick the grammar. `.tsx` needs the dedicated TSX grammar; plain `.ts`
    would mis-parse JSX, and the JS grammar already accepts `.jsx`."""
    ext = ext.lower()
    if ext == ".tsx":
        return _TSX_LANGUAGE
    if ext in _TS_EXTENSIONS:
        return _TS_LANGUAGE
    return _JS_LANGUAGE


class TreeSitterJsParser:
    name = "tree_sitter_js"
    # Structural parser — owns code.* edges, so its writes are tagged 'ast'
    # (the only source allowed into the edges table besides agent/user).
    source_tag = "ast"
    languages = [".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"]

    def parse(self, path: Path, content: str) -> ParseResult:
        language_label = "typescript" if path.suffix.lower() in _TS_EXTENSIONS else "javascript"
        module_name = path.stem
        module_unit = ParsedUnit(
            kind="code.module",
            name=module_name,
            line_start=1,
            line_end=max(content.count("\n") + 1, 1),
            content=content,
            language=language_label,
        )

        if not AVAILABLE:
            # Defensive: the registry shouldn't route here when unavailable,
            # but never crash if it does.
            return ParseResult(units=[module_unit], language=language_label)

        try:
            source = content.encode("utf-8")
            parser = _TSParser(_language_for(path.suffix))
            tree = parser.parse(source)
        except Exception as exc:
            logger.debug("tree-sitter parse failed for %s: %s", path, exc)
            return ParseResult(units=[module_unit], language=language_label)

        walker = _Walker(
            module_name=module_name,
            source=source,
            language_label=language_label,
        )
        walker.walk(tree.root_node)

        return ParseResult(
            units=[module_unit] + walker.units,
            edges=walker.edges,
            language=language_label,
        )


class _Walker:
    """Single-pass tree walk producing units + edges, tracking the class
    and caller context as it descends — the tree-sitter analogue of the
    Python parser's `_Visitor`."""

    def __init__(self, *, module_name: str, source: bytes, language_label: str) -> None:
        self.module_name = module_name
        self.source = source
        self.language_label = language_label
        self.units: list[ParsedUnit] = []
        self.edges: list[ParsedEdge] = []
        self._class_stack: list[str] = []
        # Caller context. Module-level calls attribute to the module;
        # functions/methods push their qualified name on entry.
        self._caller_stack: list[str] = [module_name]

    def walk(self, node) -> None:
        for child in node.named_children:
            self._dispatch(child)

    def _dispatch(self, node) -> None:
        t = node.type
        if t in _CLASS_TYPES:
            self._visit_class(node)
        elif t in _FUNCTION_TYPES:
            self._visit_function(node)
        elif t == "method_definition":
            self._visit_method(node)
        elif t in _IMPORT_TYPES:
            self._visit_import(node)
        elif t in _DECL_TYPES:
            # `const f = () => {}` / `const C = class {}` — descend into the
            # declarators so a named binding becomes a function/class unit.
            self._visit_declaration(node)
        elif t == "call_expression":
            self._visit_call(node)
            self.walk(node)
        else:
            self.walk(node)

    # ── classes ────────────────────────────────────────────────────
    def _visit_class(self, node, *, name_override: str | None = None) -> None:
        name = name_override or self._field_name(node)
        if not name:
            # Anonymous class with no binding — still descend for calls.
            self.walk(node)
            return

        qualified = self._qualified_name(name)
        parent = self._current_qualified_parent()

        self.units.append(
            ParsedUnit(
                kind="code.class",
                name=qualified,
                parent_name=parent,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                content=self._node_text(node),
                language=self.language_label,
            )
        )

        for base in self._extends_targets(node):
            self.edges.append(
                ParsedEdge(
                    source_name=qualified,
                    target_name=base,
                    relation="inherits",
                    line=node.start_point[0] + 1,
                )
            )

        self._class_stack.append(name)
        body = node.child_by_field_name("body")
        if body is not None:
            self.walk(body)
        self._class_stack.pop()

    # ── functions & methods ────────────────────────────────────────
    def _visit_function(self, node, *, name_override: str | None = None) -> None:
        name = name_override or self._field_name(node)
        if not name:
            # Anonymous function expression with no binding — descend for calls.
            self.walk(node)
            return

        qualified = self._qualified_name(name)
        parent = self._current_qualified_parent()
        kind = "code.method" if self._class_stack else "code.function"

        self.units.append(
            ParsedUnit(
                kind=kind,
                name=qualified,
                parent_name=parent,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                content=self._node_text(node),
                language=self.language_label,
            )
        )

        self._caller_stack.append(qualified)
        self.walk(node)
        self._caller_stack.pop()

    def _visit_method(self, node) -> None:
        name = self._field_name(node)
        if not name:
            self.walk(node)
            return

        qualified = self._qualified_name(name)
        parent = self._current_qualified_parent()

        self.units.append(
            ParsedUnit(
                kind="code.method",
                name=qualified,
                parent_name=parent,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                content=self._node_text(node),
                language=self.language_label,
            )
        )

        self._caller_stack.append(qualified)
        body = node.child_by_field_name("body")
        if body is not None:
            self.walk(body)
        self._caller_stack.pop()

    # ── declarations: const/let/var bindings ───────────────────────
    def _visit_declaration(self, node) -> None:
        for declarator in node.named_children:
            if declarator.type != "variable_declarator":
                continue
            name_node = declarator.child_by_field_name("name")
            value = declarator.child_by_field_name("value")
            name = self._node_text(name_node) if name_node is not None else None

            if value is not None and value.type in _FUNCTION_TYPES and name:
                self._visit_function(value, name_override=name)
            elif value is not None and value.type in _CLASS_TYPES and name:
                self._visit_class(value, name_override=name)
            else:
                # Not a function/class binding — still descend for any calls
                # in the initializer (e.g. `const x = factory()`).
                self.walk(declarator)

    # ── imports ────────────────────────────────────────────────────
    def _visit_import(self, node) -> None:
        # ES `import ... from 'spec'` → the `string` child holds the specifier.
        for child in node.named_children:
            if child.type == "string":
                spec = self._string_value(child)
                if spec:
                    self.edges.append(
                        ParsedEdge(
                            source_name=self.module_name,
                            target_name=spec,
                            relation="imports",
                            line=node.start_point[0] + 1,
                        )
                    )

    # ── calls ──────────────────────────────────────────────────────
    def _visit_call(self, node) -> None:
        fn = node.child_by_field_name("function")
        if fn is None:
            return

        # `require('mod')` is an import in disguise — attribute to the module.
        if fn.type == "identifier" and self._node_text(fn) == "require":
            spec = self._require_specifier(node)
            if spec:
                self.edges.append(
                    ParsedEdge(
                        source_name=self.module_name,
                        target_name=spec,
                        relation="imports",
                        line=node.start_point[0] + 1,
                    )
                )
            return

        target = self._callable_name(fn)
        if target:
            self.edges.append(
                ParsedEdge(
                    source_name=self._caller_stack[-1],
                    target_name=target,
                    relation="calls",
                    line=node.start_point[0] + 1,
                )
            )

    # ── helpers ────────────────────────────────────────────────────
    def _qualified_name(self, leaf: str) -> str:
        return ".".join(self._class_stack + [leaf])

    def _current_qualified_parent(self) -> str | None:
        if not self._class_stack:
            return None
        return ".".join(self._class_stack)

    def _field_name(self, node) -> str | None:
        name_node = node.child_by_field_name("name")
        return self._node_text(name_node) if name_node is not None else None

    def _extends_targets(self, node) -> list[str]:
        """Names following `extends`. JS holds the identifier directly under
        `class_heritage`; TS wraps it in an `extends_clause`. `implements`
        clauses are interfaces, not `inherits` edges, so we skip them."""
        heritage = next((c for c in node.named_children if c.type == "class_heritage"), None)
        if heritage is None:
            return []
        candidates = []
        for c in heritage.named_children:
            if c.type == "extends_clause":
                candidates.extend(c.named_children)
            elif c.type == "implements_clause":
                continue
            else:
                candidates.append(c)
        return [
            self._node_text(c) for c in candidates if c.type in ("identifier", "member_expression")
        ]

    def _callable_name(self, fn) -> str | None:
        """Flatten the callee to a dotted name. `foo` -> 'foo';
        `obj.method` -> 'obj.method'. Returns None for shapes we can't
        cheaply name (computed access, IIFEs, …)."""
        if fn.type in ("identifier", "member_expression"):
            return self._node_text(fn)
        return None

    def _require_specifier(self, call_node) -> str | None:
        args = call_node.child_by_field_name("arguments")
        if args is None:
            return None
        for arg in args.named_children:
            if arg.type == "string":
                return self._string_value(arg)
        return None

    def _string_value(self, string_node) -> str | None:
        """Unwrap a `string` node to its inner text (without quotes)."""
        for c in string_node.named_children:
            if c.type == "string_fragment":
                return self._node_text(c)
        # Empty string literal ('' / "") has no fragment child.
        raw = self._node_text(string_node)
        return raw[1:-1] if len(raw) >= 2 else raw

    def _node_text(self, node) -> str:
        return self.source[node.start_byte : node.end_byte].decode("utf-8", "replace")


_CLASS_TYPES = {"class_declaration", "abstract_class_declaration", "class"}
_FUNCTION_TYPES = {
    "function_declaration",
    "generator_function_declaration",
    "function_expression",
    "generator_function",
    "arrow_function",
}
_IMPORT_TYPES = {"import_statement"}
_DECL_TYPES = {"lexical_declaration", "variable_declaration"}

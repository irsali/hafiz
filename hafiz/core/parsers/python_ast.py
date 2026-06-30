"""Python AST parser — stdlib `ast`-based.

Emits these unit kinds:
    code.module     one per file
    code.class      top-level and nested classes
    code.function   top-level functions
    code.method     functions defined inside a class

And these edge relations:
    calls           caller function/method/module -> callee
    imports         module -> imported name
    inherits        class -> base class

Qualified names follow Python convention: `OuterClass.InnerClass.method`.
On SyntaxError, degrades gracefully to a single `code.module` unit —
never crashes the ingest pipeline.
"""

from __future__ import annotations

import ast
from pathlib import Path

from hafiz.core.parsers import ParsedEdge, ParsedUnit, ParseResult


class PythonAstParser:
    name = "python_ast"
    # Structural parser — owns code.* edges, tagged 'ast' in the store.
    source_tag = "ast"
    languages = [".py"]

    def parse(self, path: Path, content: str) -> ParseResult:
        module_name = path.stem
        module_unit = ParsedUnit(
            kind="code.module",
            name=module_name,
            line_start=1,
            line_end=max(content.count("\n") + 1, 1),
            content=content,
            language="python",
        )

        try:
            tree = ast.parse(content, filename=str(path))
        except SyntaxError:
            return ParseResult(units=[module_unit], language="python")

        lines = content.splitlines(keepends=True)
        visitor = _Visitor(module_name=module_name, lines=lines)
        visitor.visit(tree)

        return ParseResult(
            units=[module_unit] + visitor.units,
            edges=visitor.edges,
            language="python",
        )


class _Visitor(ast.NodeVisitor):
    def __init__(self, *, module_name: str, lines: list[str]) -> None:
        self.module_name = module_name
        self.lines = lines
        self.units: list[ParsedUnit] = []
        self.edges: list[ParsedEdge] = []
        self._class_stack: list[str] = []
        # Caller context. Starts with the module so module-level calls
        # attribute to it; functions/methods push onto it.
        self._caller_stack: list[str] = [module_name]

    # ── units: classes ─────────────────────────────────────────────
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified = self._qualified_name(node.name)
        parent = self._current_qualified_parent()

        self.units.append(
            ParsedUnit(
                kind="code.class",
                name=qualified,
                parent_name=parent,
                line_start=node.lineno,
                line_end=getattr(node, "end_lineno", None),
                content=self._slice(node.lineno, getattr(node, "end_lineno", None)),
                language="python",
            )
        )

        for base in node.bases:
            base_name = _name_from_expr(base)
            if base_name:
                self.edges.append(
                    ParsedEdge(
                        source_name=qualified,
                        target_name=base_name,
                        relation="inherits",
                        line=node.lineno,
                    )
                )

        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    # ── units: functions & methods ─────────────────────────────────
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = self._qualified_name(node.name)
        parent = self._current_qualified_parent()
        kind = "code.method" if self._class_stack else "code.function"

        self.units.append(
            ParsedUnit(
                kind=kind,
                name=qualified,
                parent_name=parent,
                line_start=node.lineno,
                line_end=getattr(node, "end_lineno", None),
                content=self._slice(node.lineno, getattr(node, "end_lineno", None)),
                language="python",
            )
        )

        self._caller_stack.append(qualified)
        self.generic_visit(node)
        self._caller_stack.pop()

    # ── edges: imports ─────────────────────────────────────────────
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.edges.append(
                ParsedEdge(
                    source_name=self.module_name,
                    target_name=alias.name,
                    relation="imports",
                    line=node.lineno,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            target = f"{module}.{alias.name}" if module else alias.name
            self.edges.append(
                ParsedEdge(
                    source_name=self.module_name,
                    target_name=target,
                    relation="imports",
                    line=node.lineno,
                )
            )

    # ── edges: calls ───────────────────────────────────────────────
    def visit_Call(self, node: ast.Call) -> None:
        target = _name_from_expr(node.func)
        if target:
            self.edges.append(
                ParsedEdge(
                    source_name=self._caller_stack[-1],
                    target_name=target,
                    relation="calls",
                    line=node.lineno,
                )
            )
        self.generic_visit(node)

    # ── helpers ────────────────────────────────────────────────────
    def _qualified_name(self, leaf: str) -> str:
        parts = self._class_stack + [leaf]
        return ".".join(parts)

    def _current_qualified_parent(self) -> str | None:
        if not self._class_stack:
            return None
        return ".".join(self._class_stack)

    def _slice(self, start: int | None, end: int | None) -> str:
        if start is None or end is None:
            return ""
        return "".join(self.lines[start - 1 : end])


def _name_from_expr(expr: ast.expr) -> str | None:
    """Flatten a Name/Attribute expression to a dotted string. Returns
    None for expressions we can't cheaply name (Calls, Subscripts, …)."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _name_from_expr(expr.value)
        return f"{base}.{expr.attr}" if base else expr.attr
    return None

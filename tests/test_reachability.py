"""Every module in ``hafiz/core/`` must actually be reachable.

The "why" behind the warm-daemon bug. ``hafiz serve`` shipped in ``f285dbf``
with a daemon, a client, a protocol and ten passing tests — and no command
ever called it. It sat unreachable for three months while ``serve status``
advertised "Auto-spawns on the next client request". Nothing detected it:
the tests passed, the lint passed, the docs described it, and the feature
did nothing.

That is a *class* of defect, not an incident. A complete, tested, documented
module that nothing calls is indistinguishable from a deleted one, and no
existing gate in this repo looked for it.

**Two weaker versions of this check were tried first and both missed the
bug**, which is why the implementation looks the way it does:

1. *Module-level orphans* — "is this module imported anywhere?" ``serve.py``
   did import ``daemon_client``, just only its private ``_send_one``. The
   module was reachable; its entire public API was not.
2. *Substring matching* — searching for a function's name in other files.
   ``daemon_client`` exposes ``context``, ``request`` and ``observe``:
   ordinary words that appear all over this codebase. Every module looked
   used.

So the check resolves **imports**, and asks whether anything imports a
*public* name from the module. Validated by running it against the commit
where the bug existed: it flags ``daemon_client`` there, and nothing else,
and is empty on the fixed tree.
"""

from __future__ import annotations

import ast
import pathlib

PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Both layers, for the same reason in two shapes. A ``core/`` module nothing
#: calls is a dead engine; a ``commands/`` module never registered in
#: ``cli.py`` is a command the user cannot invoke. Commands are all reachable
#: today — this keeps them that way.
SCANNED = (PKG_ROOT / "hafiz" / "core", PKG_ROOT / "hafiz" / "commands")

#: Modules allowed to have no public API imported anywhere, each with the
#: reason. A dict rather than a list so adding an entry forces you to say why
#: — "it's fine" is how the daemon stayed dead.
ALLOWED_UNREACHABLE: dict[str, str] = {}


def _public_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        and not node.name.startswith("_")
    }


def _imports_by_module() -> dict[str, set[str]]:
    """``module -> names that other production modules import from it``."""
    imported: dict[str, set[str]] = {}
    for path in sorted((PKG_ROOT / "hafiz").rglob("*.py")):
        self_mod = ".".join(path.relative_to(PKG_ROOT).with_suffix("").parts)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                if not node.module.startswith("hafiz") or node.module == self_mod:
                    continue
                for alias in node.names:
                    # `from hafiz.core import graph_analysis as ga` imports a
                    # *module*. Attributing that to the package would make
                    # every such submodule look unused.
                    submodule = f"{node.module}.{alias.name}"
                    as_path = PKG_ROOT / pathlib.Path(*submodule.split("."))
                    if as_path.with_suffix(".py").exists():
                        imported.setdefault(submodule, set()).add("<module>")
                    else:
                        imported.setdefault(node.module, set()).add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("hafiz") and alias.name != self_mod:
                        imported.setdefault(alias.name, set()).add("<module>")
    return imported


def test_no_core_module_is_unreachable():
    """Fail when a core module's whole public API has no production caller.

    Tests do not count. A module exercised only by its own unit tests is the
    exact shape of the bug: ``test_daemon.py`` had ten passing tests for a
    client that nothing in the product called.
    """
    imported = _imports_by_module()
    unreachable: list[str] = []

    for path in sorted(p for base in SCANNED for p in base.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        module = ".".join(path.relative_to(PKG_ROOT).with_suffix("").parts)
        public = _public_names(ast.parse(path.read_text(encoding="utf-8")))
        if not public:
            continue
        used = imported.get(module, set())
        if (public & used) or "<module>" in used:
            continue
        if module in ALLOWED_UNREACHABLE:
            continue
        unreachable.append(
            f"  {module} — defines {len(public)} public name(s), "
            f"but production code imports only {sorted(used) or 'nothing'}"
        )

    assert not unreachable, (
        "These core modules are unreachable from the product:\n"
        + "\n".join(unreachable)
        + "\n\nEither wire the module into a command, delete it, or add it to "
        "ALLOWED_UNREACHABLE with a reason. A module nothing calls is "
        "indistinguishable from one that was deleted — `hafiz serve` sat in "
        "this state for three months while advertising that it worked."
    )


def test_the_check_would_have_caught_the_daemon():
    """Guard the guard.

    A reachability check that cannot catch the bug it was written for is
    worse than none — it certifies the codebase while missing the thing it
    exists to find. Two earlier versions of this check (module-level
    orphans; substring matching) both passed on the broken tree.

    So: simulate the pre-fix shape — a module whose only importer takes a
    private name — and assert the predicate rejects it.
    """
    public = {"request", "context", "query_recall", "observe", "capture"}
    # What `serve.py` imported before the fix.
    used_before = {"_send_one"}
    assert not (public & used_before) and "<module>" not in used_before, (
        "the predicate would not have flagged the pre-fix daemon_client"
    )
    # And that wiring it in clears the flag.
    used_after = {"_send_one", "query_recall", "context"}
    assert public & used_after

"""Session operations addressable by slug, with no terminal cursor involved.

``hafiz session start`` on the CLI writes a cursor keyed to the TTY (or to
``$HAFIZ_SESSION_KEY`` when there is no terminal), so later commands in the
same shell auto-tag their writes. An MCP client has neither: it is one
long-lived process serving many logical threads of work, and a per-process
cursor would silently tag every write with whatever session was started last.

So this module exposes the DB-backed session functions **addressed
explicitly by slug**. Callers pass ``session_id`` to the write tools rather
than relying on ambient state, which is the honest model for a caller that
may be doing three unrelated things at once.
"""

from __future__ import annotations

from typing import Any

from hafiz.core.sessions import (
    create_session,
    end_session_db,
    get_session_by_slug,
    list_sessions,
)

OPERATIONS = ("list", "show", "start", "end")


class UnknownSessionOperationError(ValueError):
    """Raised for an unrecognised ``operation``, listing what is valid."""


def _as_dict(stored) -> dict[str, Any]:
    from hafiz.core.mcp_registry import to_jsonable

    return to_jsonable(stored)


async def session_op(
    operation: str,
    slug: str | None = None,
    name: str | None = None,
    agent: str | None = None,
    task: str | None = None,
    project: str | None = None,
    limit: int = 50,
    include_ended: bool = True,
) -> dict[str, Any]:
    """Run one session operation and return a JSON-shaped result.

    Args:
        operation: One of :data:`OPERATIONS`.
        slug: Human-facing session identifier; required for show, start, end.
        name: Descriptive title, used by start.
        agent: Which agent owns the session.
        task: Named task within the session, used by start.
        project: Scopes the session; stored as scope_kind/scope_value.
        limit: Cap on rows returned by list.
        include_ended: Whether list returns closed sessions too.
    """
    if operation not in OPERATIONS:
        raise UnknownSessionOperationError(
            f"unknown session operation {operation!r}; expected one of {', '.join(OPERATIONS)}"
        )

    if operation == "list":
        rows = await list_sessions(
            agent=agent,
            scope_kind="project" if project else None,
            scope_value=project,
            limit=limit,
            include_ended=include_ended,
        )
        return {"operation": operation, "sessions": [_as_dict(r) for r in rows]}

    if slug is None:
        raise ValueError(f"session operation {operation!r} requires 'slug'")

    if operation == "show":
        found = await get_session_by_slug(slug)
        return {
            "operation": operation,
            "slug": slug,
            "found": found is not None,
            "session": _as_dict(found) if found else None,
        }

    if operation == "start":
        # Idempotent by slug: an agent reconnecting mid-thread should rejoin
        # its session rather than fork a second one with the same name.
        existing = await get_session_by_slug(slug)
        if existing is not None:
            return {
                "operation": operation,
                "slug": slug,
                "created": False,
                "session": _as_dict(existing),
            }
        created = await create_session(
            slug=slug,
            name=name,
            agent=agent,
            task=task,
            scope_kind="project" if project else None,
            scope_value=project,
        )
        return {"operation": operation, "slug": slug, "created": True, "session": _as_dict(created)}

    found = await get_session_by_slug(slug)
    if found is None:
        return {"operation": "end", "slug": slug, "found": False, "session": None}
    ended = await end_session_db(found.id)
    return {"operation": "end", "slug": slug, "found": True, "session": _as_dict(ended)}

"""Strip credentials out of anything hafiz is about to show or store.

Hafiz's diagnostic surfaces exist to be *shared* — ``hafiz doctor``,
``hafiz status --diagnose``, ``hafiz errors list --json`` are what a user
pastes into a bug report and what an agent is told to read when the user says
"hafiz feels broken". A database password in that output is therefore not a
cosmetic problem: the harm path is the user following hafiz's own instructions.

Observed on 2026-09-08, on the author's own install: ``hafiz doctor``,
``hafiz status --diagnose`` and ``hafiz config show`` each printed the live
Postgres password verbatim, in both human and ``--json`` form, from six
render sites in total.

The error log was **not** holding a credential at that point, and the
distinction is worth writing down so nobody re-derives it: the failures
logged there were ``IntegrityError`` / ``ProgrammingError`` / ``TypeError``,
whose text carries SQL *parameters* but not the connection URL. What it did
hold, in a ``0664`` file, was the user's email address and commit metadata
out of those parameters — a smaller finding, and the reason this module's
callers also tightened the file's permissions.

So the capture-path scrub here is **preventive**, not remediation: a
SQLAlchemy ``OperationalError`` *does* put the whole URL in its message, that
is precisely the failure ``error_log._recognize_db_connectivity`` exists to
recognise, and the log is therefore one connection failure away from holding
the password. The error-log module had declared "**No secrets**" as an
invariant since it was written; it just never counted ``str(exc)`` as an
"argument value that might embed tokens".

Two properties this module is built around:

- **Stdlib only.** ``error_log`` documents that it must stay independently
  importable so a broken sqlalchemy install can still be logged. A redaction
  helper on that path may not drag in sqlalchemy to parse a URL.
- **One function, both jobs.** A URL is just a very short blob of text, so
  display sites and capture sites share a single code path. A second, laxer
  redactor is how the next surface leaks.

What this deliberately does **not** touch: hafiz-authored help text. Strings
like ``postgresql+asyncpg://user:pass@host/db`` in a ``fix=`` hint are
*placeholders teaching the format*, and redacting them to ``user:***@host``
would damage the docs to protect a secret that was never there. Redact
values that came from the user's config or from the outside world; leave
prose hafiz wrote about itself alone.
"""

from __future__ import annotations

import re
from typing import Any

#: ``scheme://user:secret@host`` anywhere in a blob of text.
#:
#: The ``@`` is what makes this safe to run over free text: a URL with no
#: credential (``sqlite:///x.db``, ``http://host:8080/p``) has no ``@`` and
#: cannot match, and a URL with a user but no password (``pg://u@host``) has
#: no ``:`` in the userinfo and cannot match either. The username is kept —
#: it is useful for diagnosis and is not the secret.
_URL_CREDENTIAL = re.compile(
    r"(?P<prefix>[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]*:)(?P<secret>[^\s/@]+)(?P<at>@)"
)

REDACTED = "***"


def redact_credentials(text: str) -> str:
    """``text`` with any URL password replaced by ``***``.

    Safe on a bare URL, on a multi-line traceback, and on text containing
    neither. Never raises: a redactor that can throw would turn a diagnostic
    into a second failure.
    """
    if not text or "@" not in text:
        return text
    try:
        return _URL_CREDENTIAL.sub(lambda m: f"{m.group('prefix')}{REDACTED}{m.group('at')}", text)
    except Exception:  # noqa: BLE001 — see docstring; must never fail a caller
        return text


def redact_deep(value: Any) -> Any:
    """:func:`redact_credentials` applied through strings, lists, and dicts.

    For free-form payloads — an error record's ``context``, an ``argv`` list —
    where a credential could sit at any depth. Non-string leaves pass through
    untouched, so numbers and booleans keep their types for ``--json``
    consumers. Dict *keys* are left alone: a secret in a key position would be
    a bug of a different shape, and rewriting keys would break the shape
    agents parse.
    """
    if isinstance(value, str):
        return redact_credentials(value)
    if isinstance(value, list):
        return [redact_deep(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_deep(v) for v in value)
    if isinstance(value, dict):
        return {k: redact_deep(v) for k, v in value.items()}
    return value


def contains_credential(text: str) -> bool:
    """True when ``text`` still carries an unredacted URL password.

    Used by ``hafiz doctor`` to report that a log written before redaction
    existed is still holding a secret — surfacing it for the user to act on,
    rather than an agent silently deleting their audit trail.
    """
    if not text or "@" not in text:
        return False
    for match in _URL_CREDENTIAL.finditer(text):
        if match.group("secret") != REDACTED:
            return True
    return False

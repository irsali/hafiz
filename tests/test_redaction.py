"""Credentials must not reach anything hafiz prints, stores, or hands an agent.

The observed leak: `hafiz doctor`, `hafiz status --diagnose` and `hafiz config
show` each printed the author's live Postgres password verbatim, human and
`--json` alike, from six render sites. Those are the commands whose output a
user pastes into a bug report and an agent is told to read — so the harm path
is following hafiz's own instructions.

The error-log half is **preventive**, and the difference is recorded here
deliberately. The log inspected that day held no credential: its failures were
`IntegrityError` / `ProgrammingError` / `TypeError`, whose text carries SQL
*parameters* but not the connection URL. (An initial reading claimed otherwise
on the strength of a substring that turned out to be a git author name, not the
password.) What makes the scrub necessary anyway is that a SQLAlchemy
`OperationalError` *does* embed the whole URL in its message — the exact
failure `error_log._recognize_db_connectivity` exists for — so any log is one
connection failure away from holding it, arriving inside `str(exc)` rather than
as anything the declared "No secrets" invariant would recognise as an argument.

Two properties, tested from both ends:

  1. A secret never reaches a display surface or the log on disk.
  2. Hafiz's own `user:pass@host` *placeholders* survive — they teach the URL
     format, and redacting them would damage the advice while protecting
     nothing.
"""

from __future__ import annotations

import json
import os
import re
import stat

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core import error_log
from hafiz.core.redaction import (
    REDACTED,
    contains_credential,
    redact_credentials,
    redact_deep,
)

runner = CliRunner()

SECRET = "noble-wave-local-db-18"
LIVE_URL = f"postgresql+asyncpg://postgres:{SECRET}@localhost:5432/hafiz"


# ─── the redactor ───────────────────────────────────────────────────────


class TestRedactCredentials:
    def test_a_password_is_replaced(self):
        out = redact_credentials(LIVE_URL)
        assert SECRET not in out
        assert out == "postgresql+asyncpg://postgres:***@localhost:5432/hafiz"

    def test_the_username_survives(self):
        """It is useful for diagnosis and is not the secret."""
        assert "postgres" in redact_credentials(LIVE_URL)

    def test_the_host_and_database_survive(self):
        out = redact_credentials(LIVE_URL)
        assert "localhost:5432" in out
        assert out.endswith("/hafiz")

    @pytest.mark.parametrize(
        "safe",
        [
            "sqlite:///home/u/.local/share/hafiz/hafiz.db",
            "sqlite+aiosqlite:////abs/path.db",
            "http://localhost:8080/some/path",
            "postgresql://user@localhost/db",  # user, no password
            "no urls here at all",
            "",
        ],
    )
    def test_urls_without_a_credential_are_untouched(self, safe):
        assert redact_credentials(safe) == safe

    def test_a_port_is_not_mistaken_for_a_password(self):
        """`host:8080` has a colon but no `@`, so the early-exit guard covers it."""
        assert redact_credentials("http://example.com:8080/x") == "http://example.com:8080/x"

    def test_a_port_survives_next_to_an_unrelated_at_sign(self):
        """Now the early guard cannot help — the regex's `@` anchor has to.

        Mutation-found: dropping the anchor left this case redacting `:8080`
        as if it were a credential, and the test above could not see it
        because it exits before the regex ever runs.
        """
        text = "mail user@example.com about http://example.com:8080/x"
        assert redact_credentials(text) == text

    def test_it_works_inside_a_traceback(self):
        blob = (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in <module>\n'
            "sqlalchemy.exc.OperationalError: (asyncpg) connection failed\n"
            f"[SQL: SELECT 1]\n(Background on this error at: {LIVE_URL})\n"
        )
        out = redact_credentials(blob)
        assert SECRET not in out
        assert "OperationalError" in out
        assert "SELECT 1" in out

    def test_every_occurrence_is_replaced(self):
        out = redact_credentials(f"connecting to {LIVE_URL}; retrying {LIVE_URL}")
        assert SECRET not in out
        assert out.count(REDACTED) == 2

    def test_a_special_character_password_is_replaced(self):
        weird = "postgresql://u:p%40ss+w0rd!@host/db"
        assert redact_credentials(weird) == "postgresql://u:***@host/db"

    def test_it_is_idempotent(self):
        once = redact_credentials(LIVE_URL)
        assert redact_credentials(once) == once

    def test_a_hafiz_placeholder_is_masked_but_teaches_nothing_wrong(self):
        """A placeholder *would* be masked, which is why callers never pass one.

        The boundary is enforced at the call sites — `detail=` gets the real
        URL and is redacted, `fix=` keeps its placeholder and is not passed
        through here at all. This test states the reason that split exists.
        """
        assert redact_credentials("postgresql+asyncpg://user:pass@host/db") == (
            "postgresql+asyncpg://user:***@host/db"
        )


class TestRedactDeep:
    def test_it_walks_lists_and_dicts(self):
        payload = {"argv": ["migrate-backend", LIVE_URL], "ctx": {"url": LIVE_URL}}
        out = redact_deep(payload)
        assert SECRET not in json.dumps(out)

    def test_non_strings_keep_their_type(self):
        """`--json` consumers depend on numbers staying numbers."""
        out = redact_deep({"n": 3, "ok": True, "none": None, "f": 1.5})
        assert out == {"n": 3, "ok": True, "none": None, "f": 1.5}

    def test_dict_keys_are_left_alone(self):
        out = redact_deep({LIVE_URL: "value"})
        assert list(out) == [LIVE_URL]


class TestContainsCredential:
    def test_it_finds_a_live_secret(self):
        assert contains_credential(LIVE_URL) is True

    def test_an_already_redacted_url_is_not_a_finding(self):
        """Otherwise the doctor check would report its own fix as a failure."""
        assert contains_credential(redact_credentials(LIVE_URL)) is False

    def test_clean_text_is_not_a_finding(self):
        assert contains_credential("sqlite:///x.db") is False


# ─── the error log, both ends ───────────────────────────────────────────


@pytest.fixture
def log_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path / "hafiz" / "errors.log"


def _operational_error() -> Exception:
    """Shaped like the connection failure that *would* leak — see module docstring."""
    return RuntimeError(
        f"(asyncpg.InvalidPasswordError) password authentication failed\n"
        f"[SQL: SELECT 1]\n(Background on this error at: {LIVE_URL})"
    )


class TestErrorLogWritePath:
    def test_the_secret_never_lands_on_disk(self, log_home):
        error_log.log_exception(_operational_error(), argv=["migrate-backend", LIVE_URL])
        raw = log_home.read_text()
        assert SECRET not in raw
        assert REDACTED in raw

    def test_the_diagnosis_survives_redaction(self, log_home):
        """Scrubbing must not cost us the information the log exists for."""
        error_log.log_exception(_operational_error(), argv=["migrate-backend"])
        rec = error_log.tail(limit=1)[0]
        assert "password authentication failed" in rec.message
        assert "SELECT 1" in rec.traceback
        assert rec.command == "migrate-backend"

    # These assert on `build_record` rather than on `tail()`. Reading a record
    # back redacts it again, so a `tail()`-based assertion cannot tell a clean
    # write from a dirty write cleaned up on read — a mutation dropping the
    # write-path scrub passed both of these until they were rewritten.

    def test_argv_is_scrubbed_before_it_is_written(self, log_home):
        record = error_log.build_record(RuntimeError("boom"), argv=["migrate-backend", LIVE_URL])
        assert SECRET not in json.dumps(record.argv)

    def test_context_is_scrubbed_before_it_is_written(self, log_home, monkeypatch):
        monkeypatch.setattr(
            error_log, "_suggest_action", lambda exc, **_kw: (None, {"url": LIVE_URL})
        )
        record = error_log.build_record(RuntimeError("boom"), argv=["status"])
        assert SECRET not in json.dumps(record.context)

    def test_no_field_of_a_written_record_holds_the_secret(self, log_home, monkeypatch):
        """The catch-all: whatever the field list grows to, none of it may leak."""
        monkeypatch.setattr(
            error_log, "_suggest_action", lambda exc, **_kw: (None, {"url": LIVE_URL})
        )
        record = error_log.build_record(_operational_error(), argv=["migrate-backend", LIVE_URL])
        assert SECRET not in json.dumps(record.as_jsonable())

    def test_a_recognizer_still_sees_the_unredacted_text(self, log_home, monkeypatch):
        """Recognizers match on error wording; redacting before them could break one.

        Spies on ``traceback_text`` specifically. Spying on ``str(exc)`` would
        prove nothing — the exception object is never modified, so that would
        hold however the ordering changed.
        """
        seen = {}

        def spy(exc, *, argv, traceback_text=""):
            seen["tb"] = traceback_text
            seen["argv"] = list(argv)
            return None, {}

        monkeypatch.setattr(error_log, "_suggest_action", spy)
        error_log.build_record(_operational_error(), argv=["migrate-backend", LIVE_URL])
        assert SECRET in seen["tb"]
        assert SECRET in seen["argv"][1]

    def test_the_log_is_owner_only(self, log_home):
        error_log.log_exception(RuntimeError("boom"), argv=["status"])
        assert stat.S_IMODE(log_home.stat().st_mode) == 0o600

    def test_a_pre_existing_world_readable_log_is_tightened(self, log_home):
        """The author's own log was 0664; hardening only on create would miss it."""
        log_home.parent.mkdir(parents=True, exist_ok=True)
        log_home.write_text("")
        os.chmod(log_home, 0o664)
        error_log.log_exception(RuntimeError("boom"), argv=["status"])
        assert stat.S_IMODE(log_home.stat().st_mode) == 0o600


class TestErrorLogReadPath:
    """Records written *before* redaction existed must stop leaking too."""

    def _write_legacy_record(self, log_home):
        log_home.parent.mkdir(parents=True, exist_ok=True)
        log_home.write_text(
            json.dumps(
                {
                    "id": "abcd1234-0000-0000-0000-000000000000",
                    "timestamp": "2026-06-11T11:20:32+00:00",
                    "command": "ingest",
                    "argv": ["ingest", LIVE_URL],
                    "exception_type": "OperationalError",
                    "message": f"could not connect to {LIVE_URL}",
                    "traceback": f"Traceback...\n{LIVE_URL}\n",
                    "cwd": "/home/u",
                    "hafiz_version": "0.1.0",
                    "git_branch": "dev",
                    "git_dirty": False,
                    "host_fingerprint": "deadbeef",
                    "suggested_action": None,
                    "context": {"url": LIVE_URL},
                }
            )
            + "\n"
        )

    def test_a_legacy_record_is_redacted_on_read(self, log_home):
        self._write_legacy_record(log_home)
        rec = error_log.tail(limit=1)[0]
        assert SECRET not in json.dumps(rec.as_jsonable())
        assert "could not connect" in rec.message

    def test_errors_list_json_does_not_leak(self, log_home):
        self._write_legacy_record(log_home)
        result = runner.invoke(app, ["errors", "list", "--json"])
        assert result.exit_code == 0
        assert SECRET not in result.output

    def test_errors_show_does_not_leak(self, log_home):
        self._write_legacy_record(log_home)
        result = runner.invoke(app, ["errors", "show", "abcd1234", "--json"])
        assert result.exit_code == 0
        assert SECRET not in result.output


class TestDoctorErrorLogCheck:
    """Is the log safe to hand someone? Two one-time states a new log can't reach.

    Content: a record written before redaction existed still holds a
    credential on disk even though `errors list` no longer shows it.
    Permissions: a log created before hardening stays group/world readable
    until the next error is logged — the state the author's own install was
    actually in, holding their email address in SQL parameters.

    Both are reported, never fixed: the log is the user's audit trail.
    """

    def _find(self, payload, name="Error log is private"):
        return next(c for c in payload["checks"] if c["name"] == name)

    def _legacy(self, log_home, *, mode=0o600):
        TestErrorLogReadPath()._write_legacy_record(log_home)
        os.chmod(log_home, mode)

    def test_a_legacy_credential_fails_the_check(self, log_home):
        self._legacy(log_home)
        check = self._find(json.loads(runner.invoke(app, ["doctor", "--json"]).output))
        assert check["passed"] is False
        assert "1 record(s) still contain a database password" in check["detail"]

    def test_the_finding_is_a_count_not_the_secret(self, log_home):
        """A check about a leaked credential that prints one is self-defeating."""
        self._legacy(log_home)
        check = self._find(json.loads(runner.invoke(app, ["doctor", "--json"]).output))
        assert SECRET not in json.dumps(check)

    def test_the_remedy_is_named(self, log_home):
        self._legacy(log_home)
        assert (
            "hafiz errors clear"
            in self._find(json.loads(runner.invoke(app, ["doctor", "--json"]).output))["fix"]
        )

    def test_a_world_readable_log_fails_even_with_clean_content(self, log_home):
        """The state this install was in: no credential, but 0664."""
        log_home.parent.mkdir(parents=True, exist_ok=True)
        log_home.write_text("")
        os.chmod(log_home, 0o664)
        check = self._find(json.loads(runner.invoke(app, ["doctor", "--json"]).output))
        assert check["passed"] is False
        assert "readable beyond the owner" in check["detail"]
        assert "chmod 600" in check["fix"]

    def test_a_group_readable_log_also_fails(self, log_home):
        log_home.parent.mkdir(parents=True, exist_ok=True)
        log_home.write_text("")
        os.chmod(log_home, 0o640)
        check = self._find(json.loads(runner.invoke(app, ["doctor", "--json"]).output))
        assert check["passed"] is False

    def test_both_problems_are_reported_together(self, log_home):
        self._legacy(log_home, mode=0o664)
        check = self._find(json.loads(runner.invoke(app, ["doctor", "--json"]).output))
        assert "database password" in check["detail"]
        assert "beyond the owner" in check["detail"]

    def test_a_freshly_written_log_passes(self, log_home):
        error_log.log_exception(_operational_error(), argv=["migrate-backend", LIVE_URL])
        check = self._find(json.loads(runner.invoke(app, ["doctor", "--json"]).output))
        assert check["passed"] is True
        assert "0600" in check["detail"]

    def test_no_log_at_all_passes(self, log_home):
        check = self._find(json.loads(runner.invoke(app, ["doctor", "--json"]).output))
        assert check["passed"] is True
        assert check["detail"] == "no log yet"


# ─── display surfaces ───────────────────────────────────────────────────


class TestDisplaySurfaces:
    """Asserted against the *configured* URL, not an injected one.

    An earlier version of these tests set ``HAFIZ_DATABASE__URL`` and looked
    for that password in the output. The test harness overrides the URL, so
    the injected secret was never in scope and the assertions passed
    vacuously — they would have passed with no redaction at all. Deriving the
    secret from whatever is actually configured makes them real on both
    backend legs: Postgres has a password to hide, the embedded store has
    none and must be printed unchanged.
    """

    @staticmethod
    def _configured() -> tuple[str, str | None]:
        from hafiz.core.config import get_settings

        url = get_settings().database.url
        match = re.search(r"://[^:/@\s]*:([^@/\s]+)@", url)
        return url, (match.group(1) if match else None)

    def test_config_show_human_does_not_leak(self):
        url, password = self._configured()
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        if password is None:
            return  # embedded store: no credential exists to leak
        assert password not in result.output

    def test_config_show_json_does_not_leak(self):
        url, password = self._configured()
        result = runner.invoke(app, ["config", "show", "--json"])
        assert result.exit_code == 0
        if password is None:
            assert url in result.output  # nothing to redact; printed as-is
            return
        assert password not in result.output
        assert REDACTED in result.output

    def test_config_show_json_keeps_its_shape(self):
        """Redaction must not disturb the documented payload."""
        result = runner.invoke(app, ["config", "show", "--json"])
        payload = json.loads(result.output)
        assert "url" in payload["database"]
        assert "tunables" in payload

    def test_doctor_json_does_not_leak(self):
        _url, password = self._configured()
        result = runner.invoke(app, ["doctor", "--json"])
        if password is None:
            return
        assert password not in result.output

    def test_the_engine_still_receives_the_real_password(self):
        """Redaction is display-only — if it reached the connect path, nothing works."""
        url, password = self._configured()
        assert REDACTED not in url
        if password is not None:
            assert password in url

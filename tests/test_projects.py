"""One project, one name — synonyms, look-alikes, and refusing to guess.

A 2026-11 audit found ten products under two or three spellings each, and two
genuinely different repos one spelling apart (a retired ``Billing Portal`` and its
replacement ``billing-portal``). These tests pin the rules that came out of it:
declared synonyms and case/spacing variants resolve; a near-certain typo is
corrected on read; anything unclear is an error naming the candidates; a write
never guesses from a close spelling; and only declared synonyms are merged.

Every test builds its own names from a random stem, so the shared test database
(full of other tests' projects) cannot change what is "close".
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core import projects as P  # noqa: N812 — read as "P.resolve_project"
from hafiz.core.database import Annotation, close_engine, get_session_factory
from hafiz.core.projects import UnknownProjectError


@pytest.fixture(autouse=True)
async def _isolate():
    await close_engine()
    P.invalidate()
    yield
    P.invalidate()
    await close_engine()


@pytest.fixture
def stem() -> str:
    return "zq" + uuid.uuid4().hex[:6]


async def _seed(project: str, n: int = 1) -> None:
    async with get_session_factory()() as s:
        for i in range(n):
            s.add(Annotation(content=f"row {i} of {project}", kind="fact", project=project))
        await s.commit()
    P.invalidate()


async def _rows(project: str) -> int:
    async with get_session_factory()() as s:
        return (
            await s.execute(
                select(func.count()).select_from(Annotation).where(Annotation.project == project)
            )
        ).scalar()


# ── Keys ────────────────────────────────────────────────────────────────


def test_case_spaces_and_separators_share_a_key():
    assert {P.project_key(n) for n in ("Shop Front_v2", "Shop_Front_v2", "shop-front-v2")} == {
        "shop-front-v2"
    }


# ── Reading ─────────────────────────────────────────────────────────────


async def test_a_spacing_variant_reads_the_project_on_file(stem):
    await _seed(f"{stem.title()} API")
    assert await P.resolve_project(f"{stem}-api") == f"{stem.title()} API"


async def test_a_declared_synonym_reads_its_main_name(stem):
    await _seed(f"{stem} Designer")
    await P.add_alias(f"{stem}-dz", f"{stem} Designer")
    assert await P.resolve_project(f"{stem}-dz") == f"{stem} Designer"


async def test_two_projects_one_spelling_apart_are_not_guessed_between(stem):
    old, new = f"{stem.title()} Portal", f"{stem}-portal"
    await _seed(old)
    await _seed(new)
    with pytest.raises(UnknownProjectError) as err:
        await P.resolve_project(f"{stem} portal")
    assert set(err.value.candidates) == {old, new}


async def test_retiring_one_lets_the_active_project_win(stem):
    old, new = f"{stem.title()} Portal", f"{stem}-portal"
    await _seed(old)
    await _seed(new)
    await P.set_retired(old)
    assert await P.resolve_project(f"{stem} portal") == new
    assert await P.resolve_project(old) == old  # the exact name still reads it


async def test_a_near_certain_typo_is_corrected(stem):
    await _seed(f"{stem}-transactional")
    assert await P.resolve_project(f"{stem}-transactionl") == f"{stem}-transactional"


async def test_a_different_version_number_is_never_a_typo(stem):
    await _seed(f"{stem}-engine-v3")
    with pytest.raises(UnknownProjectError):
        await P.resolve_project(f"{stem}-engine-v4")


async def test_a_different_set_of_numbers_is_suggested_but_never_used(stem):
    # Caught on a live store: "shop-front-v4" was silently read as "Shop Front_v2_4".
    await _seed(f"{stem}-channel-v2-4")
    with pytest.raises(UnknownProjectError) as err:
        await P.resolve_project(f"{stem}-channel-v4")
    assert err.value.candidates[0] == f"{stem}-channel-v2-4"
    assert await P.canonical_project(f"{stem}-channel-v4") == f"{stem}-channel-v4"


async def test_an_unknown_name_is_an_error_with_suggestions(stem):
    await _seed(f"{stem}-ledger-api")
    with pytest.raises(UnknownProjectError) as err:
        await P.resolve_project(f"{stem}-ledger")
    assert err.value.candidates[0] == f"{stem}-ledger-api"
    assert "Did you mean" in str(err.value)


async def test_no_project_means_no_filter():
    assert await P.resolve_project(None) is None
    assert await P.resolve_project("  ") is None


# ── Writing ─────────────────────────────────────────────────────────────


async def test_writes_file_variants_and_synonyms_under_the_main_name(stem):
    await _seed(f"{stem.title()} API")
    await P.add_alias(f"{stem}-id", f"{stem.title()} API")
    assert await P.canonical_project(f"{stem}_api") == f"{stem.title()} API"
    assert await P.canonical_project(f"{stem}-id") == f"{stem.title()} API"


async def test_a_typo_of_an_existing_project_is_refused_unless_declared_new(stem):
    await _seed(f"{stem}-helpscout")
    with pytest.raises(UnknownProjectError):
        await P.canonical_project(f"{stem}-helpscot")
    assert await P.canonical_project(f"{stem}-helpscot", new_project=True) == f"{stem}-helpscot"


async def test_siblings_and_new_names_are_written_as_given(stem):
    await _seed(f"{stem}-notice-trunk")
    await _seed(f"{stem}-engine-v3")
    assert await P.canonical_project(f"{stem}-notice") == f"{stem}-notice"
    assert await P.canonical_project(f"{stem}-engine-v4") == f"{stem}-engine-v4"


async def test_an_observation_under_a_synonym_lands_on_the_main_name(stem):
    from hafiz.core.annotations import store_annotation

    await _seed(f"{stem} Account")
    await P.add_alias(f"{stem}-acct", f"{stem} Account")
    ann = await store_annotation("settlement runs nightly", project=f"{stem}-acct")
    assert ann.project == f"{stem} Account"


# ── Merging ─────────────────────────────────────────────────────────────


async def test_merge_previews_then_moves_only_declared_synonyms(stem):
    main, alias, lookalike = f"{stem} Web", f"{stem}-web-id", f"{stem}_web"
    await _seed(main)
    await _seed(alias, n=2)
    await _seed(lookalike)
    await P.add_alias(alias, main)

    preview = await P.merge_aliases()
    mine = [m for m in preview["moves"] if m["alias"] == alias]
    assert mine == [{"alias": alias, "project": main, "rows": {"annotations": 2}}]
    assert await _rows(alias) == 2  # a preview moves nothing

    await P.merge_aliases(apply=True)
    assert (await _rows(alias), await _rows(main)) == (0, 3)
    assert await _rows(lookalike) == 1  # looks alike, never declared: untouched


async def test_listing_flags_look_alikes_that_are_not_declared(stem):
    await _seed(f"{stem} Tool")
    await _seed(f"{stem}-tool")
    report = await P.list_projects()
    assert sorted([f"{stem} Tool", f"{stem}-tool"]) in report["look_alikes"]


# ── Filling in from the working directory ───────────────────────────────


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


async def test_the_repo_folder_names_the_project(tmp_path, stem):
    repo = _git_repo(tmp_path / f"{stem}-svc")
    (repo / "src").mkdir()
    assert await P.project_from_cwd(repo / "src") == f"{stem}-svc"


async def test_a_repo_folder_that_is_a_synonym_maps_to_the_main_name(tmp_path, stem):
    repo = _git_repo(tmp_path / f"{stem}-portal-v2")
    await _seed(f"{stem}-portal")
    await P.add_alias(f"{stem}-portal-v2", f"{stem}-portal")
    assert await P.project_from_cwd(repo) == f"{stem}-portal"


async def test_outside_a_repo_and_any_index_there_is_no_project(tmp_path, monkeypatch):
    async def nothing_indexed():
        return {}

    monkeypatch.setattr("hafiz.core.store.indexed_root_per_project", nothing_indexed)
    assert await P.project_from_cwd(tmp_path) is None


async def test_a_repo_nested_in_a_broadly_indexed_folder_is_its_own_project(
    tmp_path, stem, monkeypatch
):
    repo = _git_repo(tmp_path / "workspace" / f"{stem}-inner")

    async def workspace_indexed():
        return {f"{stem}-workspace": str(tmp_path / "workspace")}

    monkeypatch.setattr("hafiz.core.store.indexed_root_per_project", workspace_indexed)
    assert await P.project_from_cwd(repo) == f"{stem}-inner"
    assert await P.project_from_cwd(tmp_path / "workspace") == f"{stem}-workspace"


# ── Command line ────────────────────────────────────────────────────────


def test_an_unknown_project_on_query_exits_2_with_the_fix(stem):
    runner = CliRunner()
    seeded = runner.invoke(app, ["note", "x", "--project", f"{stem}-billing", "--new-project"])
    assert seeded.exit_code == 0, seeded.output
    result = runner.invoke(
        app, ["query", "anything", "--observations", "--project", f"{stem}-bill", "--json"]
    )
    assert result.exit_code == 2, result.output
    assert f"{stem}-billing" in json.loads(result.output)["error"]


def test_agent_notes_take_the_repo_and_user_notes_do_not(tmp_path, stem, monkeypatch):
    repo = _git_repo(tmp_path / f"{stem}-cli")
    monkeypatch.chdir(repo)
    runner = CliRunner()
    agent = runner.invoke(
        app, ["note", f"agent thought {stem}", "--source", "agent:test", "--json"]
    )
    user = runner.invoke(app, ["note", f"user thought {stem}", "--source", "user:test", "--json"])
    assert agent.exit_code == 0 and user.exit_code == 0, (agent.output, user.output)
    assert json.loads(agent.output)["annotation"]["project"] == f"{stem}-cli"
    assert json.loads(user.output)["annotation"]["project"] is None


def test_projects_alias_and_merge_from_the_command_line(stem):
    runner = CliRunner()
    for name in (f"{stem} Hub", f"{stem}-hub-id"):
        seeded = runner.invoke(app, ["note", f"about {name}", "--project", name, "--new-project"])
        assert seeded.exit_code == 0, seeded.output
    aliased = runner.invoke(app, ["projects", "alias", f"{stem}-hub-id", f"{stem} Hub"])
    assert aliased.exit_code == 0, aliased.output
    merged = runner.invoke(app, ["projects", "merge", "--apply", "--json"])
    assert merged.exit_code == 0, merged.output
    assert any(m["alias"] == f"{stem}-hub-id" for m in json.loads(merged.output)["moves"])
    listed = json.loads(runner.invoke(app, ["projects", "--json"]).output)
    hub = next(p for p in listed["projects"] if p["name"] == f"{stem} Hub")
    assert hub["aliases"] == [f"{stem}-hub-id"] and hub["rows"] == 2


async def test_inside_the_daemon_a_correction_is_collected_for_the_caller(stem):
    # Logged in the daemon, the notice would reach nobody who typed the name.
    await _seed(f"{stem}-transactional")
    token = P.notices.set([])
    try:
        await P.resolve_project(f"{stem}-transactionl")
        said = P.notices.get()
    finally:
        P.notices.reset(token)
    assert said and f"{stem}-transactional" in said[0]

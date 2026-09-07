"""A ~520 MB download with no output is indistinguishable from a hang.

Cold-start measurement: `hafiz init` is 0.39s, then the first `ingest` stalls
25.1s — 22 of which is fastembed fetching the embedding model, unbounded on a
slow link, with nothing from hafiz explaining the wait. Worse, it happens
*twice*: reranking is enabled by default and loads lazily, so a second ~90 MB
cross-encoder lands on the first `query --observations`, a different command
well after ingest already paid its toll. ~610 MB across two silent stalls.

The rational response to a silent multi-minute stall is Ctrl-C, which is
precisely what leaves the half-written cache `_purge_if_incomplete` exists to
repair. So the notice is not decoration — it prevents the failure mode.

Two properties this file pins hard:

  * **Nothing on stdout, ever.** `hafiz mcp` speaks JSON-RPC over stdio and
    `--json` is a documented contract. A notice on stdout would corrupt both,
    and only on a cold cache — the hardest case to reproduce.
  * **A corrupt cache counts as not-cached.** The notice and the repair path
    read one predicate. A laxer "does the directory exist?" check would report
    a corrupt cache as present and stay silent in the one case where the user
    is about to wait for a full re-download.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hafiz.core import embeddings
from hafiz.core.config import EmbeddingSettings, RerankSettings

MODEL = "nomic-ai/nomic-embed-text-v1.5"


@pytest.fixture(autouse=True)
def _clean_announce_state():
    """The notice is once-per-process, so tests must not inherit each other."""
    embeddings._announced.clear()
    yield
    embeddings._announced.clear()


def _write_cache(
    cache_dir: Path,
    model_name: str = MODEL,
    *,
    onnx: bool = True,
    incomplete: bool = False,
    idle_seconds: float | None = None,
) -> Path:
    """Build a HuggingFace-shaped model cache, optionally broken.

    ``idle_seconds`` backdates every mtime in the tree. The repair path reads
    mtimes to tell an abandoned download from one in flight, so a partial cache
    written "just now" is deliberately treated as live.
    """
    model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    snap = model_dir / "snapshots" / "deadbeefdeadbeef"
    (snap / "onnx").mkdir(parents=True)
    if onnx:
        (snap / "onnx" / "model.onnx").write_bytes(b"weights")
    blobs = model_dir / "blobs"
    blobs.mkdir(parents=True)
    if incomplete:
        (blobs / "abc123.incomplete").write_bytes(b"")
    if idle_seconds is not None:
        _backdate(model_dir, idle_seconds)
    return model_dir


def _backdate(model_dir: Path, seconds: float) -> None:
    when = time.time() - seconds
    for path in sorted(model_dir.rglob("*"), reverse=True):
        os.utime(path, (when, when))
    os.utime(model_dir, (when, when))


# ── the predicate ────────────────────────────────────────────────────


def test_a_complete_cache_reads_as_cached(tmp_path):
    _write_cache(tmp_path)
    assert embeddings.model_is_cached(tmp_path, MODEL)


def test_an_absent_cache_reads_as_not_cached(tmp_path):
    assert not embeddings.model_is_cached(tmp_path, MODEL)


def test_an_interrupted_download_reads_as_not_cached(tmp_path):
    """Guard the guard — the case the whole design turns on.

    An `*.incomplete` blob is the signature of a Ctrl-C'd download. If this
    ever reported True, the notice would go silent exactly when the user is
    about to re-wait for the full download.
    """
    _write_cache(tmp_path, incomplete=True)
    assert not embeddings.model_is_cached(tmp_path, MODEL)


def test_a_snapshot_without_weights_reads_as_not_cached(tmp_path):
    """The other corruption shape: config/tokenizer landed, model.onnx didn't."""
    _write_cache(tmp_path, onnx=False)
    assert not embeddings.model_is_cached(tmp_path, MODEL)


@pytest.mark.parametrize(
    ("onnx", "incomplete", "should_purge"),
    [
        (True, False, False),  # healthy — leave it alone
        (True, True, True),  # interrupted download
        (False, False, True),  # weights missing
        (False, True, True),  # both
    ],
)
def test_purge_removes_exactly_what_is_not_cached(tmp_path, onnx, incomplete, should_purge):
    """The repair path and the notice must never disagree about "cached".

    They share one predicate precisely so a future edit can't drift them apart;
    this pins the equivalence rather than trusting the refactor that created it.
    Trees are backdated: liveness is a separate axis, covered below.
    """
    model_dir = _write_cache(tmp_path, onnx=onnx, incomplete=incomplete, idle_seconds=3600)

    purged = embeddings._purge_if_incomplete(tmp_path, MODEL)

    assert purged is should_purge
    assert model_dir.exists() is not should_purge


# ── liveness: never delete a download someone else is doing ──────────
#
# From the directory shape alone, an in-progress download and an abandoned one
# are identical: a `blobs/*.incomplete` and no resolvable `onnx/model.onnx`.
# huggingface_hub serializes its own downloads under a lock in `<cache>/.locks`,
# but this repair path runs outside that lock, so an unconditional rmtree
# deletes the tree another process is actively writing into — corrupting the
# cache this function exists to repair. mtime is the liveness signal.


def test_a_download_in_flight_is_left_alone(tmp_path):
    """The bug: a partial cache written moments ago is someone's live download.

    The client used to spawn a daemon, time out after 20s of a ~22s download,
    fall back to direct execution, and then rmtree the daemon's half-written
    tree — producing a corrupt cache on the user's very first command.
    """
    model_dir = _write_cache(tmp_path, onnx=False, incomplete=True)

    assert embeddings._purge_if_incomplete(tmp_path, MODEL) is False
    assert model_dir.exists(), "deleting a live download is the bug, not the fix"


def test_a_download_in_flight_still_reads_as_not_cached(tmp_path):
    """Left alone, but not pretended to be usable.

    The notice must still fire and the loader must still wait — fastembed then
    blocks on the hub's own lock, which is the correct outcome.
    """
    _write_cache(tmp_path, onnx=False, incomplete=True)
    assert not embeddings.model_is_cached(tmp_path, MODEL)


def test_an_abandoned_download_is_purged(tmp_path):
    """The case the repair path exists for — self-healing must still work."""
    model_dir = _write_cache(tmp_path, onnx=False, incomplete=True, idle_seconds=3600)

    assert embeddings._purge_if_incomplete(tmp_path, MODEL) is True
    assert not model_dir.exists()


def test_activity_anywhere_in_the_tree_counts_as_live(tmp_path):
    """The narrow window before the big blob exists at all.

    A process that has created the snapshot dirs but not yet started writing
    `model.onnx` has no `*.incomplete` blob to read an mtime from, which is why
    liveness is judged across the whole tree rather than off that one file.
    """
    model_dir = _write_cache(tmp_path, onnx=False, incomplete=False, idle_seconds=3600)
    # A fresh touch deep in the tree is the only evidence of the new process.
    (model_dir / "snapshots" / "deadbeefdeadbeef" / "onnx" / "model.onnx.incomplete").write_bytes(
        b""
    )

    assert embeddings._purge_if_incomplete(tmp_path, MODEL) is False
    assert model_dir.exists()


def test_the_staleness_window_is_long_enough_to_span_a_slow_link(tmp_path):
    """It only has to exceed the gap *between writes*, not the download time.

    But a too-short window reintroduces the bug on any host that pauses, so
    pin the shipped value rather than leaving it free to be "tuned" downward.
    """
    assert embeddings._STALE_DOWNLOAD_SECONDS >= 120


def test_purge_is_a_noop_when_there_is_nothing_to_purge(tmp_path):
    assert embeddings._purge_if_incomplete(tmp_path, MODEL) is False


# ── the notice ───────────────────────────────────────────────────────


def test_the_notice_names_the_model_the_size_and_the_destination(tmp_path, capsys):
    """A wait the user can reason about: how long, how big, where it lands."""
    embeddings.announce_download(tmp_path, MODEL, purpose="embedding")

    err = capsys.readouterr().err
    assert MODEL in err
    assert "520 MB" in err, "the size is what makes the wait legible"
    assert "one time" in err
    assert str(tmp_path) in err or "~/" in err


def test_the_notice_says_where_the_bytes_come_from(tmp_path, capsys):
    """Hafiz's claim is that the store stays local; this is the one outbound call.

    A sovereign tool should name the moment it contacts a third party rather
    than let a stranger find it in a packet trace.
    """
    embeddings.announce_download(tmp_path, MODEL, purpose="embedding")

    err = capsys.readouterr().err
    assert "huggingface.co" in err
    assert "Nothing from your store is sent" in err


def test_the_notice_never_reaches_stdout(tmp_path, capsys):
    """`--json` and the MCP stdio framing both live on stdout.

    An agent that asked for JSON and got a Rich panel prepended to it simply
    fails to parse, and a JSON-RPC client desynchronizes outright.
    """
    embeddings.announce_download(tmp_path, MODEL, purpose="embedding")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err != "", "...but it must still be visible somewhere"


def test_one_wait_gets_one_notice(tmp_path, capsys):
    """The auto-device path can build a model twice: GPU probe fails → CPU.

    Both builds go through the same loader, and there is only one download.
    """
    embeddings.announce_download(tmp_path, MODEL, purpose="embedding")
    embeddings.announce_download(tmp_path, MODEL, purpose="embedding")

    assert capsys.readouterr().err.count(MODEL) == 1


def test_the_two_models_announce_independently(tmp_path, capsys):
    """Separate downloads on separate commands; deduping is per model."""
    embeddings.announce_download(tmp_path, MODEL, purpose="embedding")
    embeddings.announce_download(tmp_path, "Xenova/ms-marco-MiniLM-L-6-v2", purpose="reranker")

    err = capsys.readouterr().err
    assert MODEL in err
    assert "ms-marco" in err


def test_retry_re_arms_the_notice(tmp_path, capsys):
    """`hafiz embedding retry` exists to re-download; that wait needs a notice too."""
    embeddings.announce_download(tmp_path, MODEL, purpose="embedding")
    embeddings.reset_cache()
    embeddings.announce_download(tmp_path, MODEL, purpose="embedding")

    assert capsys.readouterr().err.count(MODEL) == 2


def test_an_unknown_model_still_warns_about_the_wait(tmp_path, capsys):
    """A user-configured model has no measured size, but the stall is real."""
    embeddings.announce_download(tmp_path, "some-org/some-model", purpose="embedding")

    err = capsys.readouterr().err
    assert "some-org/some-model" in err
    assert "MB" in err


def test_the_shipped_models_both_have_a_measured_size():
    """Read from the settings defaults, so swapping a default model fails here.

    Falling back to the vague "a few hundred MB" for a model hafiz itself ships
    is a silent downgrade of the only number that makes the wait legible.
    """
    for name in (EmbeddingSettings().model, RerankSettings().model):
        assert name in embeddings._MODEL_SIZE_HINT, (
            f"{name} is a shipped default with no measured download size; "
            "measure it with `du -sh ~/.cache/hafiz/models/models--*` and add it"
        )


# ── the wiring ───────────────────────────────────────────────────────
#
# The notice existing is not the same as the notice firing. These patch the
# model constructors so the call sites are exercised without a real download.


def test_the_embedding_loader_announces_on_a_cold_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(embeddings, "_model_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(embeddings, "TextEmbedding", lambda **kw: object())

    embeddings._text_embedding(MODEL, ["CPUExecutionProvider"])

    assert MODEL in capsys.readouterr().err


def test_the_embedding_loader_is_silent_on_a_warm_cache(tmp_path, monkeypatch, capsys):
    """The 99.9% case. A notice on every run would be noise, not information."""
    _write_cache(tmp_path)
    monkeypatch.setattr(embeddings, "_model_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(embeddings, "TextEmbedding", lambda **kw: object())

    embeddings._text_embedding(MODEL, ["CPUExecutionProvider"])

    assert capsys.readouterr().err == ""


def test_a_corrupt_cache_is_announced_too(tmp_path, monkeypatch, capsys):
    """The longest wait of all must not be the quietest.

    A corrupt cache means a full re-download, and it is the one case a naive
    "is the model directory there?" check gets backwards. Note that the
    purge/announce *order* at the call site is immaterial — swapping the two
    lines leaves every test here green — precisely because the predicate reads
    completeness rather than presence. The order is just the natural one
    (repair, then decide); it is the predicate doing the work.
    """
    _write_cache(tmp_path, incomplete=True)
    monkeypatch.setattr(embeddings, "_model_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(embeddings, "TextEmbedding", lambda **kw: object())

    embeddings._text_embedding(MODEL, ["CPUExecutionProvider"])

    assert MODEL in capsys.readouterr().err


def test_the_reranker_loader_announces_on_a_cold_cache(tmp_path, monkeypatch, capsys):
    """The download the cold-start audit missed, because it ran plain `query`."""
    import fastembed.rerank.cross_encoder as ce

    from hafiz.core import reranker

    monkeypatch.setattr(reranker, "_model_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(ce, "TextCrossEncoder", lambda **kw: object())

    reranker._build_reranker()

    err = capsys.readouterr().err
    assert RerankSettings().model in err
    assert "reranker" in err


@pytest.fixture
def spawn_probe(tmp_path, monkeypatch):
    """Neuter the real spawn and record whether it was attempted."""
    from hafiz.core import daemon_client

    calls: list[tuple] = []
    monkeypatch.setattr(daemon_client, "_SPAWN_WARMUP_TIMEOUT", 0.0)
    monkeypatch.setattr("hafiz.core.embeddings._model_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: calls.append((a, kw)))
    return calls


async def test_a_cold_cache_does_not_hand_the_download_to_the_daemon(spawn_probe, capsys):
    """The daemon cannot announce its own download, and must not race for it.

    It spawns with stderr=DEVNULL and warms the model *before* binding its
    socket, so a first-run download runs where nobody can see it while the
    client sits in a readiness poll looking hung. The poll's 20s budget is
    also shorter than a ~22s download, so it expires, the caller falls back to
    direct execution, and two processes fetch the same model into one cache.
    On a cold cache the foreground does it — visibly, and alone.
    """
    from hafiz.core import daemon_client

    await daemon_client._spawn_daemon()

    captured = capsys.readouterr()
    assert EmbeddingSettings().model in captured.err
    assert captured.out == "", "the client's stdout is still the JSON channel"
    assert spawn_probe == [], "a cold cache must not spawn a daemon to download in the dark"


async def test_a_warm_cache_spawns_the_daemon_as_before(tmp_path, spawn_probe, capsys):
    """The skip is scoped to the cold path; the warm path is the whole point of the daemon."""
    from hafiz.core import daemon_client

    _write_cache(tmp_path, EmbeddingSettings().model)

    await daemon_client._spawn_daemon()

    assert capsys.readouterr().err == ""
    assert len(spawn_probe) == 1, "a warm cache should still get a daemon"


def test_the_reranker_loader_is_silent_on_a_warm_cache(tmp_path, monkeypatch, capsys):
    import fastembed.rerank.cross_encoder as ce

    from hafiz.core import reranker

    _write_cache(tmp_path, RerankSettings().model)
    monkeypatch.setattr(reranker, "_model_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(ce, "TextCrossEncoder", lambda **kw: object())

    reranker._build_reranker()

    assert capsys.readouterr().err == ""

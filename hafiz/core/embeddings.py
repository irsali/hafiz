"""Embedding service using fastembed directly.

Uses nomic-embed-text-v1.5 (768 dims) by default, running locally via ONNX.

Device selection is a three-tier decision, in precedence order:

1. Explicit config (`embedding.device = "cpu" | "gpu" | "auto"`). Wins outright.
   Env override: ``HAFIZ_EMBEDDING__DEVICE=cpu``.
2. Sticky cache from a prior probe (`~/.cache/hafiz/device_state.json`).
   Written once per probe outcome, reused silently on subsequent runs.
   Auto-invalidated when the stamped onnxruntime version changes.
3. Probe CUDA on first use; persist the verdict.

Inspect or override via ``hafiz embedding status`` / ``hafiz embedding retry``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path

from fastembed import TextEmbedding
from rich.console import Console
from rich.panel import Panel

from hafiz.core import device_state
from hafiz.core.config import get_settings

logger = logging.getLogger(__name__)
_console = Console(stderr=True)

_embed_model: TextEmbedding | None = None


def _cuda_available() -> bool:
    """True if onnxruntime reports CUDAExecutionProvider in its provider list."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def _gpu_name() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        lines = out.decode().strip().splitlines()
        return lines[0] if lines else None
    except (subprocess.SubprocessError, OSError):
        return None


def _model_cache_dir() -> Path:
    """Persistent, XDG-aware cache for downloaded embedding models.

    fastembed's default is ``tempfile.gettempdir()/fastembed_cache``. On hosts
    where ``/tmp`` is tmpfs (RAM-backed) that means a ~260 MB re-download on
    every reboot — plus a fresh window for an interrupted download to leave the
    cache half-written. Pinning the cache under ``~/.cache/hafiz`` makes it
    survive reboots and turns model corruption into a once-ever event.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    path = root / "hafiz" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _purge_if_incomplete(cache_dir: Path, model_name: str) -> bool:
    """Delete a half-downloaded model cache so fastembed re-downloads cleanly.

    A download interrupted mid-flight leaves the small config/tokenizer blobs
    in place but the large ``model.onnx`` as a 0-byte ``*.incomplete`` blob with
    no snapshot symlink. fastembed then tries to load a file that isn't there
    and dies on every call instead of resuming. Detect that signature — a
    ``*.incomplete`` blob, or no resolvable ``onnx/model.onnx`` in any
    snapshot — and remove the model dir. Returns True if anything was purged.
    """
    model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    if not model_dir.is_dir():
        return False

    blobs = model_dir / "blobs"
    has_incomplete = blobs.is_dir() and any(blobs.glob("*.incomplete"))

    snapshots = model_dir / "snapshots"
    has_onnx = snapshots.is_dir() and any(
        (snap / "onnx" / "model.onnx").exists() for snap in snapshots.iterdir()
    )

    if has_incomplete or not has_onnx:
        shutil.rmtree(model_dir, ignore_errors=True)
        logger.warning(
            "Purged incomplete embedding-model cache at %s; re-downloading.",
            model_dir,
        )
        return True
    return False


def _text_embedding(model_name: str, providers: list[str]) -> TextEmbedding:
    """Construct a fastembed model against the persistent cache.

    Self-heals a corrupt cache before loading, and on any load failure raises
    an actionable error naming the cache dir and the remedy instead of letting
    a bare ONNX ``NO_SUCHFILE`` traceback escape.
    """
    cache_dir = _model_cache_dir()
    _purge_if_incomplete(cache_dir, model_name)
    try:
        return TextEmbedding(model_name=model_name, providers=providers, cache_dir=str(cache_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}' from {cache_dir}: "
            f"{exc}\nThe model download is likely corrupt or incomplete. Run "
            f"`hafiz embedding retry` (re-downloads on the next embed), or delete "
            f"{cache_dir} and retry to force a clean download."
        ) from exc


def _build_cpu_model(model_name: str) -> TextEmbedding:
    return _text_embedding(model_name, ["CPUExecutionProvider"])


def _tensorrt_available() -> bool:
    """True iff TensorRT is importable AND ORT's TRT EP is registered.

    Importing tensorrt is what loads libnvinfer.so into the process; without
    that side effect, ORT can't dlopen the EP at session creation. The pip
    package isn't a hafiz dependency — users opt in by installing it.
    """
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "TensorrtExecutionProvider" in ort.get_available_providers()


def _trt_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "hafiz" / "trt_engines"


def _build_gpu_model(model_name: str) -> TextEmbedding:
    """Build a GPU-preferring model and exercise it to surface lazy init failures.

    When TensorRT is installed, prefer it and skip the CUDA EP entirely:
    falling through from a missing TRT engine to a CUDA EP whose kernels lack
    support for newer compute capabilities (e.g. Blackwell sm_120) would just
    re-introduce the silent-NaN failure mode this function is guarding against.
    """
    if _tensorrt_available():
        cache = _trt_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        # ORT TRT EP reads these at session creation; setdefault leaves
        # operator-set values alone for advanced users.
        os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
        os.environ.setdefault("ORT_TENSORRT_CACHE_PATH", str(cache))
        providers = ["TensorrtExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    model = _text_embedding(model_name, providers)
    # fastembed defers ORT session creation until first use; force it now so
    # OOM / missing-kernel errors surface during probe, not mid-ingest.
    probe = next(iter(model.embed(["probe"])))
    if not all(math.isfinite(float(v)) for v in probe):
        raise RuntimeError(
            "GPU probe returned non-finite values (NaN/Inf). This onnxruntime "
            "build likely lacks kernels for your GPU's compute capability. "
            "On Blackwell (sm_120), `pip install tensorrt` enables TensorRT "
            "EP as an alternative GPU path; otherwise hafiz will use CPU."
        )
    return model


def _announce_fallback(state: device_state.DeviceState, *, first_time: bool) -> None:
    """Loud panel on fresh fallback; quiet INFO on sticky reuse."""
    if first_time:
        _console.print(
            Panel(
                f"[yellow]GPU probe failed:[/yellow] {state.reason}\n"
                f"Using [cyan]CPU[/cyan] for embeddings from now on.\n"
                f"Run [bold]hafiz embedding retry[/bold] to re-probe after fixing.",
                title="Embedding device",
                border_style="yellow",
                padding=(0, 1),
            )
        )
    else:
        logger.info(
            "Embedding model using CPU (sticky fallback, probed %s: %s)",
            state.probed_at,
            state.reason_category,
        )


def probe_device(
    device: str,
    model_name: str,
    *,
    persist: bool = True,
) -> tuple[TextEmbedding, device_state.DeviceState]:
    """Resolve ``device`` to a working model, writing sticky state if persist=True.

    ``device`` is one of ``"auto" | "cpu" | "gpu"``.
    - ``"cpu"``: build CPU model.
    - ``"gpu"``: build GPU model; raise on failure (no silent fallback).
    - ``"auto"``: try GPU if available, else CPU; loud panel on fallback.
    """
    if device == "cpu":
        model = _build_cpu_model(model_name)
        state = device_state.build_state(
            "cpu",
            reason="embedding.device = cpu",
            category=None,
            gpu_name=None,
        )
        if persist:
            device_state.save_state(state)
        return model, state

    if device == "gpu":
        if not _cuda_available():
            raise RuntimeError(
                "embedding.device = 'gpu' but CUDAExecutionProvider is not available. "
                "Install onnxruntime-gpu or set embedding.device = 'auto' / 'cpu'."
            )
        try:
            model = _build_gpu_model(model_name)
        except Exception as exc:
            raise RuntimeError(f"embedding.device = 'gpu' but probe failed: {exc}") from exc
        state = device_state.build_state("gpu", reason=None, category=None, gpu_name=_gpu_name())
        if persist:
            device_state.save_state(state)
        return model, state

    # device == "auto"
    if _cuda_available():
        try:
            model = _build_gpu_model(model_name)
        except Exception as exc:
            category, message = device_state.classify_exception(exc)
            state = device_state.build_state(
                "cpu", reason=message, category=category, gpu_name=_gpu_name()
            )
            if persist:
                device_state.save_state(state)
            _announce_fallback(state, first_time=True)
            return _build_cpu_model(model_name), state

        state = device_state.build_state("gpu", reason=None, category=None, gpu_name=_gpu_name())
        if persist:
            device_state.save_state(state)
        return model, state

    state = device_state.build_state(
        "cpu",
        reason="CUDAExecutionProvider not available in this onnxruntime build.",
        category="provider_unavailable",
        gpu_name=None,
    )
    if persist:
        device_state.save_state(state)
    return _build_cpu_model(model_name), state


def get_embed_model() -> TextEmbedding:
    """Lazy singleton; selects device per config → sticky cache → probe."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    settings = get_settings()
    configured = settings.embedding.device
    model_name = settings.embedding.model

    # Explicit config always wins; do not consult or overwrite the cache.
    if configured in ("cpu", "gpu"):
        _embed_model, _ = probe_device(configured, model_name, persist=False)
        return _embed_model

    # Auto: try sticky cache first.
    cached = device_state.load_state()
    if cached is not None and not device_state.is_stale(cached):
        if cached.device == "cpu":
            _embed_model = _build_cpu_model(model_name)
            _announce_fallback(cached, first_time=False)
            return _embed_model
        if cached.device == "gpu":
            try:
                _embed_model = _build_gpu_model(model_name)
                return _embed_model
            except Exception as exc:
                logger.warning("Cached GPU state failed re-init (%s); reprobing.", exc)
                device_state.clear_state()

    _embed_model, _ = probe_device("auto", model_name, persist=True)
    return _embed_model


# Per-call sub-batch size as a function of max_part_chars. Real ingest
# can submit hundreds of parts at once (one large file → many parts);
# without internal chunking, peak RSS scales linearly with the number
# of parts and OOMs the host. Values derived from the
# embedding.max_part_chars probe data (batch=8 peaks):
#
#     2K chars → ~2 GB at batch=8  → per-doc ~250 MB
#     4K chars → ~3.8 GB at batch=8 → per-doc ~475 MB
#     8K chars → ~10 GB at batch=8 → per-doc ~1.25 GB
#    16K chars → ~35 GB at batch=8 → per-doc ~4.3 GB  (O(n²) attention!)
#
# Picked so the per-call peak stays under ~4 GB above baseline on any
# host, regardless of how many parts the caller submits.
_SAFE_BATCH_FOR_CHARS: tuple[tuple[int, int], ...] = (
    (16_000, 1),
    (8_000, 2),
    (4_000, 8),
    (2_000, 24),
    (0, 32),
)


def _safe_batch_size(max_part_chars: int) -> int:
    for threshold, batch in _SAFE_BATCH_FOR_CHARS:
        if max_part_chars >= threshold:
            return batch
    return 8  # unreachable; the (0, ...) row catches everything


async def _embed_batch(model: TextEmbedding, texts: list[str]) -> list[list[float]]:
    """Run fastembed's sync ``embed`` in a worker thread, return plain Python floats."""
    arrays = await asyncio.to_thread(lambda: list(model.embed(texts)))
    return [[float(v) for v in arr] for arr in arrays]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning a list of vectors.

    Chunks internally so peak RSS doesn't scale with ``len(texts)``.
    Real ingest can submit hundreds of parts in one call (large file
    → many parts at the configured ``embedding.max_part_chars``);
    handing all of them to fastembed at once peaks RSS proportional
    to the part count and OOMs the host. Sub-batch size is picked
    from ``max_part_chars`` so the per-call peak is bounded
    regardless of how big the caller's list is.
    """
    if not texts:
        return []
    model = get_embed_model()

    # Resolve through the tunable layer (env > toml > sticky > default)
    # so a user's `hafiz config set` takes effect at the embed call.
    from hafiz.core.tunables import resolve as resolve_tunable

    max_chars = resolve_tunable("embedding.max_part_chars")
    sub_batch = _safe_batch_size(max_chars)

    if len(texts) <= sub_batch:
        return await _embed_batch(model, texts)

    out: list[list[float]] = []
    for i in range(0, len(texts), sub_batch):
        chunk = texts[i : i + sub_batch]
        out.extend(await _embed_batch(model, chunk))
    return out


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = get_embed_model()
    arrays = await asyncio.to_thread(lambda: list(model.query_embed([query])))
    return [float(v) for v in arrays[0]]


def reset_cache() -> None:
    """Drop the in-process singleton. Used by tests and `hafiz embedding retry`."""
    global _embed_model
    _embed_model = None

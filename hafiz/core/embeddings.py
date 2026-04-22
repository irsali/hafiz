"""Embedding service wrapping llama-index-embeddings-fastembed.

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

import logging
import subprocess

from llama_index.embeddings.fastembed import FastEmbedEmbedding
from rich.console import Console
from rich.panel import Panel

from hafiz.core import device_state
from hafiz.core.config import get_settings

logger = logging.getLogger(__name__)
_console = Console(stderr=True)

_embed_model: FastEmbedEmbedding | None = None


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


def _build_cpu_model(model_name: str) -> FastEmbedEmbedding:
    return FastEmbedEmbedding(model_name=model_name, providers=["CPUExecutionProvider"])


def _build_gpu_model(model_name: str) -> FastEmbedEmbedding:
    """Build a GPU-preferring model and exercise it to surface lazy init failures."""
    model = FastEmbedEmbedding(
        model_name=model_name,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    # FastEmbed defers ORT session creation until first use; force it now so
    # CUDA OOM / missing-kernel errors surface during probe, not mid-ingest.
    model.get_text_embedding("probe")
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
) -> tuple[FastEmbedEmbedding, device_state.DeviceState]:
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
        state = device_state.build_state(
            "gpu", reason=None, category=None, gpu_name=_gpu_name()
        )
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

        state = device_state.build_state(
            "gpu", reason=None, category=None, gpu_name=_gpu_name()
        )
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


def get_embed_model() -> FastEmbedEmbedding:
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
                logger.warning(
                    "Cached GPU state failed re-init (%s); reprobing.", exc
                )
                device_state.clear_state()

    _embed_model, _ = probe_device("auto", model_name, persist=True)
    return _embed_model


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning a list of vectors."""
    model = get_embed_model()
    embeddings = await model.aget_text_embedding_batch(texts)
    return embeddings


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = get_embed_model()
    return await model.aget_query_embedding(query)


def reset_cache() -> None:
    """Drop the in-process singleton. Used by tests and `hafiz embedding retry`."""
    global _embed_model
    _embed_model = None

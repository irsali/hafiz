"""Per-tunable probers — the measurement side of self-tuning.

Each prober takes a :class:`~hafiz.core.host_probe.HostProbe` snapshot
and returns a :class:`~hafiz.core.tunables.ProbeResult` recommending a
value for its tunable, with rationale + confidence.

Design constraints:

  - **Probes must not OOM the host.** This is the literal bug the
    tunable registry was built to fix. Heavy embed calls run in a
    subprocess so a bad candidate can only kill the subprocess.
  - **Probes must mirror real workload shape.** ``store.py`` collects
    all parts of all changed revisions in a file and submits them in a
    single ``embed_fn(all_part_texts)`` call, so peak RSS scales with
    the batch size — not the per-document size. The prober embeds
    multiple same-size docs in one call to reflect that. (The original
    single-doc probe under-estimated by ~order-of-magnitude and
    licensed values that OOM-killed a desktop session — see the
    decision recorded on 2026-04-27.)
  - **Probes must be bounded in time.** Every subprocess call has a
    hard timeout; we accept partial results rather than hang a
    ``hafiz doctor --probe`` session.
  - **Conservative fallback.** When we can't complete a probe (no
    fastembed, subprocess crashed, timeout), recommend the built-in
    default — never something more aggressive.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from typing import Any

from hafiz.core.host_probe import HostProbe
from hafiz.core.tunables import ProbeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# embedding.max_part_chars
# ---------------------------------------------------------------------------

# Ordered ascending: we stop at the first candidate that blows the budget,
# and the largest candidate that fit becomes the recommendation.
_MAX_PART_CHARS_CANDIDATES = [2_000, 4_000, 8_000, 16_000]

# Per-candidate subprocess timeout. A failing big candidate hangs ONNX for
# a while; cap that pain.
_PROBE_TIMEOUT_S = 180

# How many same-size docs we embed in a single ``model.embed(...)`` call
# during the probe. Mirrors how store.py batches parts of a single file:
# typical real-world batch is dozens of parts, so 8 is a conservative
# middle ground that still surfaces the multi-batch peak without blowing
# the probe's own memory budget on hosts that genuinely can't handle the
# largest candidate.
_PROBE_BATCH_SIZE = 8

# Fraction of *available* RAM we're willing to use as headroom. Available
# is volatile (drops when the user opens VSCode / browser / dev servers
# after probing), so we ALSO cap by total RAM — see _compute_budget below.
_RAM_AVAILABLE_FRACTION = 0.30

# Fraction of *total* RAM we're willing to use. This is the stable anchor:
# a recommendation must survive normal day-to-day RAM pressure, not just
# the lucky moment the probe ran. Tighter than the available fraction
# because total >> available on a working desktop.
_RAM_TOTAL_FRACTION = 0.15

# Hard ceiling for any non-GPU recommendation. ONNX attention is O(n²) in
# sequence length; even on a 128 GB workstation, 16K chars × batched parts
# can spike the working set hard enough to degrade interactive workloads.
# Users who want bigger parts can `hafiz config set` it explicitly.
_CPU_CEILING_CHARS = 8_000

# GPU shortcut thresholds. We require *total* VRAM as the gate, not free,
# because free VRAM is even more volatile than free RAM (any
# GPU-accelerated app — IDE, browser compositor, video call — moves it).
# A recommendation that's safe at probe time but unsafe when the user
# resumes their normal apps is exactly the failure mode we're fixing.
_GPU_VRAM_TOTAL_MB_FOR_8K = 16_000   # 16 GB-class card (e.g. RTX 4080, 5060 Ti)
_GPU_VRAM_TOTAL_MB_FOR_16K = 24_000  # 24 GB-class card (e.g. RTX 3090, 4090)
_GPU_VRAM_FREE_MB_REQUIRED = 6_000   # leave headroom for compositor + apps


# Subprocess script — embedded as a string. Imports are intentionally
# scoped inside main() so the import-error path can be reported via JSON.
_MEASURE_SCRIPT = r"""
import json, os, resource, sys, time
from pathlib import Path


def _model_cache_dir():
    # Mirror hafiz.core.embeddings._model_cache_dir without importing the
    # module — keeps this measurement subprocess's baseline RSS clean. Must
    # stay in sync so the probe reuses the same persistent model download.
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    path = root / "hafiz" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

def main():
    cfg = json.loads(sys.stdin.read())
    model_name = cfg["model"]
    candidates = cfg["candidates"]
    device = cfg.get("device", "cpu")
    batch = cfg.get("batch_size", 1)
    # Hard ceiling for THIS subprocess's RSS — set by the parent so a
    # measurement run can never push the host into swap thrash.
    # Predictive: we extrapolate the next candidate's likely peak from
    # the previous candidate (peak scales roughly linearly with chars
    # for fixed batch on this model) and skip if it would exceed the
    # ceiling. The ceiling is enforced before measurement, not by the
    # OOM killer after the fact.
    safety_ceiling_mb = cfg.get("safety_ceiling_mb")
    # Empirical scale-up factor between adjacent candidate sizes when
    # the candidates double (2K→4K→8K→16K). Measured on nomic-embed
    # under batch=8: typical ratio is 2.5-3.5x. Use 3.0 as a
    # middle-ground predictor; any candidate predicted to peak above
    # the safety ceiling is skipped without being measured.
    scale_factor = cfg.get("scale_factor", 3.0)

    try:
        from fastembed import TextEmbedding
    except Exception as e:
        print(json.dumps({"_fatal": f"fastembed unavailable: {e!s}"}), flush=True)
        sys.exit(2)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "gpu"
        else ["CPUExecutionProvider"]
    )
    try:
        model = TextEmbedding(model_name, providers=providers, cache_dir=_model_cache_dir())
    except Exception as e:
        print(json.dumps({"_fatal": f"model load failed: {e!s}"}), flush=True)
        sys.exit(2)

    last_peak_mb = None
    for c in candidates:
        # ── Predictive safety brake ────────────────────────────────
        # Skip this candidate if its likely peak (extrapolated from the
        # previous measurement) would exceed the safety ceiling. This is
        # the brake that prevents the probe from OOM-ing the host on the
        # way up to the largest candidate.
        if last_peak_mb is not None and safety_ceiling_mb is not None:
            predicted = last_peak_mb * scale_factor
            if predicted > safety_ceiling_mb:
                print(
                    json.dumps(
                        {
                            "_skipped": (
                                f"candidate {c} predicted ~{int(predicted)} MB "
                                f"(prev {last_peak_mb} MB × {scale_factor}) > "
                                f"safety ceiling {safety_ceiling_mb} MB"
                            )
                        }
                    ),
                    flush=True,
                )
                break

        text = ("lorem ipsum dolor sit amet consectetur " * ((c // 40) + 1))[:c]
        # Real ingest submits all parts of a file in one embed call. Mirror
        # that with a batch of identical-size docs.
        texts = [text] * batch
        t0 = time.time()
        try:
            _ = list(model.embed(texts, batch_size=batch))
            peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux ru_maxrss is kB; macOS reports bytes. Detect by
            # implausibility: if the number is huge for a small process,
            # assume bytes.
            peak_mb = peak_kb // 1024 if peak_kb < 100_000_000 else peak_kb // (1024 * 1024)
            result = {
                "chars": c,
                "batch": batch,
                "peak_rss_mb": peak_mb,
                "wall_s": round(time.time() - t0, 2),
                "ok": True,
            }
            last_peak_mb = peak_mb
        except Exception as e:
            result = {
                "chars": c,
                "batch": batch,
                "peak_rss_mb": None,
                "wall_s": round(time.time() - t0, 2),
                "ok": False,
                "error": str(e)[:300],
            }
        print(json.dumps(result), flush=True)
        if not result["ok"]:
            # Don't try larger candidates after a failure; they'll fail too
            # and potentially OOM the subprocess before reporting.
            break
        # ── Reactive safety brake ──────────────────────────────────
        # Even if the prediction was too optimistic, stop walking up
        # the moment we've measured a peak that already exceeds the
        # safety ceiling. The next candidate would be strictly worse.
        if (
            safety_ceiling_mb is not None
            and result["peak_rss_mb"] is not None
            and result["peak_rss_mb"] > safety_ceiling_mb
        ):
            print(
                json.dumps(
                    {
                        "_stopped": (
                            f"candidate {c} peaked {result['peak_rss_mb']} MB > "
                            f"safety ceiling {safety_ceiling_mb} MB"
                        )
                    }
                ),
                flush=True,
            )
            break

main()
"""


def _run_measurement(
    candidates: list[int],
    *,
    model: str,
    device: str,
    timeout: int,
    batch_size: int,
    safety_ceiling_mb: int | None = None,
) -> list[dict[str, Any]]:
    """Call the embedded subprocess script; return whatever it emitted.

    On timeout or non-zero exit, returns the partial results that did
    make it to stdout (subprocess flushes per-candidate), plus an
    appended ``_incomplete`` sentinel row so callers can report accurately.

    ``safety_ceiling_mb`` caps the subprocess's allowed peak RSS — the
    subprocess refuses to measure any candidate whose extrapolated peak
    would exceed the ceiling, and stops walking the moment a measured
    peak crosses it. This is what keeps the probe itself from OOM-ing
    the host on a memory-tight box.
    """
    cfg = {
        "model": model,
        "candidates": candidates,
        "device": device,
        "batch_size": batch_size,
        "safety_ceiling_mb": safety_ceiling_mb,
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _MEASURE_SCRIPT],
            input=json.dumps(cfg),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        raw = proc.stdout
        incomplete_reason = None if proc.returncode == 0 else (
            proc.stderr.strip()[:300] or f"exit {proc.returncode}"
        )
    except subprocess.TimeoutExpired as e:
        raw = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        incomplete_reason = f"timeout after {timeout}s"

    results: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if incomplete_reason:
        results.append({"_incomplete": incomplete_reason})
    return results


def _compute_budget(host: HostProbe) -> tuple[int, str]:
    """Return ``(budget_mb, basis)`` for the CPU recommendation path.

    Two anchors, take the **min**:

      - *Available* RAM × 30%: the comfortable headroom at probe time.
      - *Total* RAM × 15%: the stable anchor that survives day-to-day
        load (IDE + browser + dev servers reduce available far below
        what the probe saw).

    Floored at 1500 MB so a transient available-dip doesn't force
    everyone onto the smallest candidate.
    """
    available = host.ram_available_mb or 8_000
    total = host.ram_total_mb or available

    avail_budget = int(available * _RAM_AVAILABLE_FRACTION)
    total_budget = int(total * _RAM_TOTAL_FRACTION)

    if avail_budget <= total_budget:
        return max(1_500, avail_budget), f"30% of {available} MB available"
    return (
        max(1_500, total_budget),
        f"15% of {total} MB total (capped — available={available} MB)",
    )


def probe_max_part_chars(host: HostProbe) -> ProbeResult:
    """Recommend the largest safe ``embedding.max_part_chars`` for this host.

    Strategy:

      1. **GPU shortcut (banded).** A 24 GB+ card with ≥6 GB free can
         license 16 K. A 16 GB-class card with ≥6 GB free gets 8 K.
         Anything smaller falls through to the CPU path. Free VRAM is
         too volatile to be the sole gate — we also require sufficient
         *total* VRAM so the recommendation survives the user opening a
         GPU-accelerated app after running ``config apply``.
      2. **CPU path (batched).** Compute the budget as
         ``min(available × 30%, total × 15%)``; run candidates ascending
         in a subprocess that embeds a batch of ``_PROBE_BATCH_SIZE``
         same-size docs in one call (mirroring how ``store.py`` actually
         batches parts of a file). Recommend the largest candidate that
         stayed under budget, capped at ``_CPU_CEILING_CHARS``.
      3. **Conservative fallback.** If measurement can't run (no
         fastembed, subprocess failure), recommend the built-in default
         (2 000) — never something more aggressive.
    """
    # ── 1. GPU shortcut ───────────────────────────────────────────────
    gpu_total = host.gpu_vram_total_mb or 0
    gpu_free = host.gpu_vram_free_mb or 0
    if gpu_total and gpu_free >= _GPU_VRAM_FREE_MB_REQUIRED:
        if gpu_total >= _GPU_VRAM_TOTAL_MB_FOR_16K:
            return ProbeResult(
                recommended_value=16_000,
                rationale=(
                    f"GPU with {gpu_total} MB VRAM total / {gpu_free} MB free — "
                    f"24 GB-class card; 16K chars (~4K tokens) is well within "
                    f"ONNX CUDA working set. Skipping measurement."
                ),
                confidence="medium",
                measured={
                    "path": "gpu_shortcut_24gb",
                    "gpu_vram_total_mb": gpu_total,
                    "gpu_vram_free_mb": gpu_free,
                    "gpu_name": host.gpu_name,
                },
            )
        if gpu_total >= _GPU_VRAM_TOTAL_MB_FOR_8K:
            return ProbeResult(
                recommended_value=8_000,
                rationale=(
                    f"GPU with {gpu_total} MB VRAM total / {gpu_free} MB free — "
                    f"16 GB-class card; 8K chars (~2K tokens) leaves enough "
                    f"VRAM headroom for the compositor and other GPU apps. "
                    f"Manually raise to 16000 if you don't share the GPU."
                ),
                confidence="medium",
                measured={
                    "path": "gpu_shortcut_16gb",
                    "gpu_vram_total_mb": gpu_total,
                    "gpu_vram_free_mb": gpu_free,
                    "gpu_name": host.gpu_name,
                },
            )

    # ── 2. CPU measurement ────────────────────────────────────────────
    budget_mb, budget_basis = _compute_budget(host)

    # Safety ceiling for the probe SUBPROCESS itself. We refuse to let
    # the measurement push past the recommendation budget — there's no
    # value in measuring a candidate we already know we won't recommend.
    # An earlier version of this prober walked all candidates ascending
    # without this brake; on a 64 GB host the 16K-char × batch=8
    # candidate peaked at 35 GB RSS, swap-thrashed the desktop, and
    # made the IDE unresponsive. The brake stops that.
    safety_ceiling_mb = budget_mb

    measurements = _run_measurement(
        _MAX_PART_CHARS_CANDIDATES,
        model="nomic-ai/nomic-embed-text-v1.5",
        device="cpu",
        timeout=_PROBE_TIMEOUT_S,
        batch_size=_PROBE_BATCH_SIZE,
        safety_ceiling_mb=safety_ceiling_mb,
    )

    ok_rows = [
        r for r in measurements
        if r.get("ok") and isinstance(r.get("peak_rss_mb"), int)
    ]

    # ── 3. Conservative fallback ──────────────────────────────────────
    if not ok_rows:
        reason = "subprocess produced no usable measurements"
        for r in measurements:
            if "_fatal" in r:
                reason = r["_fatal"]
                break
            if "_incomplete" in r:
                reason = r["_incomplete"]
                break
        return ProbeResult(
            recommended_value=2_000,
            rationale=(
                f"Probe could not measure — staying on the CPU-safe default. "
                f"Reason: {reason}"
            ),
            confidence="low",
            measured={
                "path": "fallback",
                "budget_mb": budget_mb,
                "budget_basis": budget_basis,
                "raw": measurements,
            },
        )

    # Pick the largest candidate whose peak stayed under budget.
    best = 2_000
    best_row: dict[str, Any] | None = None
    for r in ok_rows:
        if r["peak_rss_mb"] <= budget_mb:
            best = r["chars"]
            best_row = r
        else:
            break

    if best_row is None:
        # Even the smallest candidate exceeded budget — host is *very*
        # memory-tight. Stick to the default, but flag it.
        return ProbeResult(
            recommended_value=2_000,
            rationale=(
                f"Even the smallest candidate (2_000 chars) peaked above the "
                f"{budget_mb} MB budget ({budget_basis}). "
                f"Keep the default and close other memory-heavy apps before "
                f"re-probing."
            ),
            confidence="low",
            measured={
                "path": "budget_exceeded",
                "budget_mb": budget_mb,
                "budget_basis": budget_basis,
                "candidates": ok_rows,
            },
        )

    # Apply the global ceiling. Even if 16K chars fit the budget on a
    # huge box, recommending it from `config apply` is too aggressive —
    # users who want it can set it explicitly.
    capped = min(best, _CPU_CEILING_CHARS)
    if capped < best:
        rationale = (
            f"Largest part size with batched peak RSS ({best_row['peak_rss_mb']} MB, "
            f"batch={_PROBE_BATCH_SIZE}) under the {budget_mb} MB budget "
            f"({budget_basis}) was {best}; capped at {capped} per the CPU-path "
            f"ceiling. Set `embedding.max_part_chars` explicitly to override."
        )
    else:
        rationale = (
            f"Largest part size with batched peak RSS ({best_row['peak_rss_mb']} MB, "
            f"batch={_PROBE_BATCH_SIZE}) under the {budget_mb} MB budget "
            f"({budget_basis})."
        )

    return ProbeResult(
        recommended_value=capped,
        rationale=rationale,
        confidence="high",
        measured={
            "path": "cpu_measured",
            "batch_size": _PROBE_BATCH_SIZE,
            "budget_mb": budget_mb,
            "budget_basis": budget_basis,
            "uncapped_best": best,
            "ceiling": _CPU_CEILING_CHARS,
            "candidates": ok_rows,
        },
    )

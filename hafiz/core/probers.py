"""Per-tunable probers — the measurement side of self-tuning.

Each prober takes a :class:`~hafiz.core.host_probe.HostProbe` snapshot
and returns a :class:`~hafiz.core.tunables.ProbeResult` recommending a
value for its tunable, with rationale + confidence.

Design constraints:

  - **Probes must not OOM the host.** This is the literal bug the
    tunable registry was built to fix. Heavy embed calls run in a
    subprocess so a bad candidate can only kill the subprocess.
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
_PROBE_TIMEOUT_S = 120

# Fraction of available RAM we're willing to use as headroom for a
# single-batch embed. Keeps room for the rest of the pipeline, DB session,
# OS, and whatever else is running on the host. Tuned against measured
# peaks: 30% is comfortable, 50% starts to risk swap on small laptops.
_RAM_BUDGET_FRACTION = 0.30

# Subprocess script — embedded as a string. Imports are intentionally
# scoped inside main() so the import-error path can be reported via JSON.
_MEASURE_SCRIPT = r"""
import json, os, resource, sys, time

def main():
    cfg = json.loads(sys.stdin.read())
    model_name = cfg["model"]
    candidates = cfg["candidates"]
    device = cfg.get("device", "cpu")

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
        model = TextEmbedding(model_name, providers=providers)
    except Exception as e:
        print(json.dumps({"_fatal": f"model load failed: {e!s}"}), flush=True)
        sys.exit(2)

    for c in candidates:
        text = ("lorem ipsum dolor sit amet consectetur " * ((c // 40) + 1))[:c]
        t0 = time.time()
        try:
            _ = list(model.embed([text], batch_size=1))
            peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux ru_maxrss is kB; macOS reports bytes. Detect by
            # implausibility: if the number is huge for a small process,
            # assume bytes.
            peak_mb = peak_kb // 1024 if peak_kb < 100_000_000 else peak_kb // (1024 * 1024)
            result = {
                "chars": c,
                "peak_rss_mb": peak_mb,
                "wall_s": round(time.time() - t0, 2),
                "ok": True,
            }
        except Exception as e:
            result = {
                "chars": c,
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

main()
"""


def _run_measurement(
    candidates: list[int], *, model: str, device: str, timeout: int
) -> list[dict[str, Any]]:
    """Call the embedded subprocess script; return whatever it emitted.

    On timeout or non-zero exit, returns the partial results that did
    make it to stdout (subprocess flushes per-candidate), plus an
    appended ``_incomplete`` sentinel row so callers can report accurately.
    """
    cfg = {"model": model, "candidates": candidates, "device": device}
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
        # e.stdout is bytes | str depending on ``text=True``; normalize.
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


def probe_max_part_chars(host: HostProbe) -> ProbeResult:
    """Recommend the largest safe ``embedding.max_part_chars`` for this host.

    Strategy:

      1. **GPU shortcut.** When a GPU has ≥ 8 GB of VRAM currently free,
         we skip measurement and recommend 16 000. Probing ONNX CUDA
         memory precisely is fiddly and the savings don't justify the
         complexity — 16 K chars is known-safe on consumer GPUs.
      2. **CPU path.** Compute a budget as 30% of available RAM; run
         candidates ascending in a subprocess, record each peak RSS;
         recommend the largest candidate that stayed under budget.
      3. **Conservative fallback.** If measurement can't run (no
         fastembed, subprocess failure), recommend the built-in default
         (2 000).
    """
    # ── 1. GPU shortcut ───────────────────────────────────────────────
    if host.gpu_vram_free_mb and host.gpu_vram_free_mb >= 8_000:
        return ProbeResult(
            recommended_value=16_000,
            rationale=(
                f"GPU with {host.gpu_vram_free_mb} MB VRAM free — 16K chars "
                f"(~4K tokens) is well within ONNX CUDA working set on "
                f"consumer GPUs. Skipping measurement."
            ),
            confidence="medium",
            measured={
                "path": "gpu_shortcut",
                "gpu_vram_free_mb": host.gpu_vram_free_mb,
                "gpu_name": host.gpu_name,
            },
        )

    # ── 2. CPU measurement ────────────────────────────────────────────
    available_mb = host.ram_available_mb or 8_000
    # Floor the budget so we don't recommend 2_000 on a host that genuinely
    # has enough RAM just because available_mb dipped transiently.
    budget_mb = max(1_500, int(available_mb * _RAM_BUDGET_FRACTION))

    measurements = _run_measurement(
        _MAX_PART_CHARS_CANDIDATES,
        model="nomic-ai/nomic-embed-text-v1.5",
        device="cpu",
        timeout=_PROBE_TIMEOUT_S,
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
                "ram_available_mb": available_mb,
                "budget_mb": budget_mb,
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
                f"{budget_mb} MB budget — host is memory-constrained. "
                f"Keep the default and close other memory-heavy apps before "
                f"re-probing."
            ),
            confidence="low",
            measured={
                "path": "budget_exceeded",
                "ram_available_mb": available_mb,
                "budget_mb": budget_mb,
                "candidates": ok_rows,
            },
        )

    return ProbeResult(
        recommended_value=best,
        rationale=(
            f"Largest part size with peak RSS ({best_row['peak_rss_mb']} MB) "
            f"under the {budget_mb} MB budget "
            f"({int(_RAM_BUDGET_FRACTION * 100)}% of {available_mb} MB "
            f"available)."
        ),
        confidence="high",
        measured={
            "path": "cpu_measured",
            "ram_available_mb": available_mb,
            "budget_mb": budget_mb,
            "candidates": ok_rows,
        },
    )

"""Offline Calibration & Gameplay Replay Benchmark Pipeline for ForeverNavigator.

Replays recorded developer gameplay logs (`data/dev_gameplay/run_*`) and synthetic
ground-truth crops sampled from cached WoW: Forever maps through `MinimapTracker` to:
  1. Produce a compact, token-efficient diagnostic scorecard (`data/calibration/latest_report.md` & `.json`)
  2. Measure zone stability, ping-pong switches, false dungeon jumps, confidence, and latency
  3. Benchmark exact pixel/yard localization error on synthetic CASC ground-truth crops
  4. Optionally sweep tracker calibration parameters (`--tune`) locally on CPU/GPU

Usage:
    python -m src.navigator.calibration_pipeline
    python -m src.navigator.calibration_pipeline --run run_20261126_125532 --stride 1
    python -m src.navigator.calibration_pipeline --synthetic 40
    python -m src.navigator.calibration_pipeline --tune --stride 4
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .minimap_tracker import MinimapROI, MinimapTracker
from .tracker import LightweightDungeonStore
from .world_maps import WorldMapRegistry


@dataclass
class RunMetrics:
    run_name: str
    frames: int
    mean_conf: float
    p10_conf: float
    mean_ms: float
    p95_ms: float
    zone_switches: int
    ping_pong_switches: int
    false_dungeon_jumps: int
    dominant_zones: List[Tuple[str, int]] = field(default_factory=list)
    worst_anomalies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SyntheticBenchMetrics:
    samples: int
    zone_accuracy: float
    median_err_yd: float
    p90_err_yd: float
    mean_ms: float


def discover_gameplay_runs(dev_root: Path = Path("data/dev_gameplay")) -> List[Path]:
    """Return all recorded gameplay run directories that contain at least one minimap PNG."""
    if not dev_root.exists():
        return []
    runs = []
    for p in sorted(dev_root.glob("run_*")):
        mm_dir = p / "minimap"
        if mm_dir.is_dir() and any(mm_dir.glob("*.png")):
            runs.append(p)
    return runs


def build_tracker_from_workspace(
    index_dir: Path = Path("data/index"),
    cache_dir: Path = Path("data/cache/maps"),
    high_precision: bool = False,
    neural_model_name: Optional[str] = None,
) -> Tuple[MinimapTracker, WorldMapRegistry]:
    """Instantiate MinimapTracker with the exact same indexed + cached maps as the live app."""
    store = LightweightDungeonStore(index_dir)
    reg = WorldMapRegistry(index_dir, cache_dir)
    for entry in reg.list_all_maps():
        if entry.cached and entry.slug not in store.dungeons and entry.category != "Continents":
            try:
                d = reg.load_map(entry.slug, preview_only=True)
                store.register_dungeon(d)
            except Exception:
                pass
    tracker = MinimapTracker(
        store.dungeons,
        store.dungeon_order,
        roi=MinimapROI(),
        high_precision=high_precision,
        lazy_precompute=True,
        neural_model_name=neural_model_name,
    )
    return tracker, reg


def replay_gameplay_run(
    run_dir: Path,
    tracker: MinimapTracker,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> RunMetrics:
    """Replay a recorded run's minimap PNGs chronologically through MinimapTracker."""
    tracker.reset()
    mm_files = sorted((run_dir / "minimap").glob("*.png"))[:: max(1, stride)]
    if max_frames is not None:
        mm_files = mm_files[:max_frames]

    confs: List[float] = []
    latencies: List[float] = []
    slug_seq: List[str] = []
    zone_counts: Dict[str, int] = {}
    anomalies: List[Dict[str, Any]] = []
    zone_switches = 0
    ping_pong = 0
    false_dungeon_jumps = 0
    recent_black = 0

    for idx, fpath in enumerate(mm_files):
        bgr = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        is_black = float((rgb.mean(axis=-1) > 18.0).mean()) < 0.05
        if is_black:
            recent_black = 5
        else:
            recent_black = max(0, recent_black - 1)

        est = tracker.update(rgb, is_already_cropped=True)
        confs.append(float(est.position_confidence))
        latencies.append(float(est.inference_ms))
        slug = est.dungeon_slug
        zone_counts[slug] = zone_counts.get(slug, 0) + 1

        if slug_seq and slug != slug_seq[-1]:
            zone_switches += 1
            prev_slug = slug_seq[-1]
            # Ping-pong: switched away and right back within 4 frames (A -> B -> A)
            if len(slug_seq) >= 2 and slug in slug_seq[-4:-1]:
                ping_pong += 1
            prev_outdoor = tracker._is_outdoor_zone.get(prev_slug, False)
            cur_outdoor = tracker._is_outdoor_zone.get(slug, False)
            if prev_outdoor and not cur_outdoor and recent_black == 0:
                false_dungeon_jumps += 1

        if est.position_confidence < 0.40 and not is_black:
            anomalies.append({
                "file": f"{run_dir.name}/minimap/{fpath.name}",
                "idx": idx,
                "slug": slug,
                "conf": round(float(est.position_confidence), 3),
                "map_px": list(est.map_px),
            })
        slug_seq.append(slug)

    confs_arr = np.array(confs, dtype=np.float64) if confs else np.zeros(1)
    lat_arr = np.array(latencies, dtype=np.float64) if latencies else np.zeros(1)
    dom = sorted(zone_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    anomalies.sort(key=lambda a: a["conf"])

    return RunMetrics(
        run_name=run_dir.name,
        frames=len(confs),
        mean_conf=round(float(confs_arr.mean()), 4),
        p10_conf=round(float(np.percentile(confs_arr, 10)), 4),
        mean_ms=round(float(lat_arr.mean()), 2),
        p95_ms=round(float(np.percentile(lat_arr, 95)), 2),
        zone_switches=zone_switches,
        ping_pong_switches=ping_pong,
        false_dungeon_jumps=false_dungeon_jumps,
        dominant_zones=dom,
        worst_anomalies=anomalies[:5],
    )


def run_synthetic_ground_truth_bench(
    tracker: MinimapTracker,
    n_samples: int = 30,
    seed: int = 42,
) -> SyntheticBenchMetrics:
    """Sample known-(x, y) synthetic minimap crops from cached/indexed maps with realistic perturbations."""
    rng = np.random.default_rng(seed)
    candidates = [s for s, d in tracker.dungeons.items() if d.map_image.shape[0] >= 256]
    if not candidates:
        return SyntheticBenchMetrics(0, 0.0, 0.0, 0.0, 0.0)

    hits = 0
    errs_yd: List[float] = []
    lats: List[float] = []
    tested = 0

    for _ in range(n_samples):
        slug = str(rng.choice(candidates))
        d = tracker.dungeons[slug]
        img = d.map_image
        h, w = img.shape[:2]
        ratio = float(d.map_size_px) / float(h)
        # Pick a random walkable/non-black center inside bounds_xy if available
        lum = img.mean(axis=-1)
        valid_mask = lum > 30.0
        if d.bounds_xy is not None:
            xmin, ymin, xmax, ymax = d.bounds_xy
            p0 = d.world_to_map_px(np.array([xmin, ymin]))
            p1 = d.world_to_map_px(np.array([xmax, ymax]))
            bx0, bx1 = max(0, int(min(p0[0], p1[0]) / ratio)), min(w, int(max(p0[0], p1[0]) / ratio))
            by0, by1 = max(0, int(min(p0[1], p1[1]) / ratio)), min(h, int(max(p0[1], p1[1]) / ratio))
            bmask = np.zeros_like(valid_mask)
            bmask[by0:by1, bx0:bx1] = True
            valid_mask = valid_mask & bmask

        r_yd = float(rng.choice([62.0, 95.0, 140.0, 200.0]))
        half_px = max(14, int(round(r_yd * d.map_scale / ratio)))
        if half_px * 2 >= min(h, w) - 8:
            continue
        valid_mask[:half_px, :] = False
        valid_mask[-half_px:, :] = False
        valid_mask[:, :half_px] = False
        valid_mask[:, -half_px:] = False
        ys, xs = np.nonzero(valid_mask)
        if len(xs) == 0:
            continue
        pick = int(rng.integers(0, len(xs)))
        cx, cy = int(xs[pick]), int(ys[pick])
        crop = img[cy - half_px : cy + half_px, cx - half_px : cx + half_px].copy()
        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_LINEAR)
        # Simulate slight brightness/contrast shift + small player arrow in center
        gain = float(rng.uniform(0.90, 1.10))
        crop = np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        cv2.circle(crop, (80, 80), 4, (240, 220, 90), -1)

        tracker.reset()
        est = tracker.update(crop, is_already_cropped=True)
        lats.append(float(est.inference_ms))
        tested += 1
        true_wx, true_wy = d.map_px_to_world(cx * ratio, cy * ratio)
        if est.dungeon_slug == slug:
            hits += 1
            err = float(np.hypot(float(est.world_pos[0]) - true_wx, float(est.world_pos[1]) - true_wy))
            errs_yd.append(err)

    if tested == 0:
        return SyntheticBenchMetrics(0, 0.0, 0.0, 0.0, 0.0)
    err_arr = np.array(errs_yd, dtype=np.float64) if errs_yd else np.array([999.0])
    return SyntheticBenchMetrics(
        samples=tested,
        zone_accuracy=round(hits / float(tested), 4),
        median_err_yd=round(float(np.median(err_arr)), 1),
        p90_err_yd=round(float(np.percentile(err_arr, 90)), 1),
        mean_ms=round(float(np.mean(lats)), 2),
    )


def format_compact_markdown(report: Dict[str, Any]) -> str:
    """Render a concise (~30-line) markdown scorecard for fast human & AI inspection."""
    lines = [
        "# ForeverNavigator Calibration Report",
        f"- Generated: `{report['generated_at']}` | Mode: `{report['mode']}` | Total runtime: `{report['elapsed_sec']:.1f}s`",
        "",
        "## Recorded Gameplay Replay",
        "| Run | Frames | Mean Conf | P10 Conf | Switches | Ping-Pong | False Dungeon | Mean ms | Top Zones |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report.get("runs", []):
        top_z = ", ".join(f"{z}({n})" for z, n in r["dominant_zones"][:3])
        lines.append(
            f"| `{r['run_name']}` | {r['frames']} | {r['mean_conf']:.2f} | {r['p10_conf']:.2f} | "
            f"{r['zone_switches']} | {r['ping_pong_switches']} | {r['false_dungeon_jumps']} | "
            f"{r['mean_ms']:.1f} | {top_z} |"
        )
    syn = report.get("synthetic_ground_truth")
    if syn and syn.get("samples", 0) > 0:
        lines.extend([
            "",
            "## Synthetic CASC Ground-Truth Benchmark",
            f"- Samples: `{syn['samples']}` | Zone Accuracy: **`{syn['zone_accuracy']*100:.1f}%`** | "
            f"Median Error: **`{syn['median_err_yd']:.1f} yd`** | P90 Error: `{syn['p90_err_yd']:.1f} yd` | "
            f"Mean Latency: `{syn['mean_ms']:.1f} ms`",
        ])
    worst = []
    for r in report.get("runs", []):
        worst.extend(r.get("worst_anomalies", [])[:2])
    if worst:
        worst.sort(key=lambda a: a["conf"])
        lines.extend(["", "## Top Anomaly Frames to Inspect"])
        for a in worst[:6]:
            lines.append(f"- `{a['file']}` -> `{a['slug']}` (conf={a['conf']:.3f}, px={a['map_px']})")
    return "\n".join(lines) + "\n"


def run_calibration(
    dev_root: Path = Path("data/dev_gameplay"),
    out_dir: Path = Path("data/calibration"),
    run_filter: Optional[str] = None,
    stride: int = 2,
    synthetic_samples: int = 25,
    high_precision: bool = False,
    neural_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    tracker, _reg = build_tracker_from_workspace(
        high_precision=high_precision,
        neural_model_name=neural_model_name,
    )
    runs = discover_gameplay_runs(dev_root)
    if run_filter:
        runs = [r for r in runs if run_filter in r.name]

    run_results = [asdict(replay_gameplay_run(r, tracker, stride=stride)) for r in runs]
    syn_res = asdict(run_synthetic_ground_truth_bench(tracker, n_samples=synthetic_samples)) if synthetic_samples > 0 else None
    tracker.close()

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "high_precision" if high_precision else "standard",
        "stride": stride,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "runs": run_results,
        "synthetic_ground_truth": syn_res,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = format_compact_markdown(report)
    (out_dir / "latest_report.md").write_text(md, encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="ForeverNavigator Offline Calibration & Replay Pipeline")
    ap.add_argument("--run", type=str, default=None, help="Filter to a specific run folder name (substring)")
    ap.add_argument("--stride", type=int, default=2, help="Frame stride across recorded minimaps (1=every frame)")
    ap.add_argument("--synthetic", type=int, default=25, help="Number of synthetic CASC ground-truth samples")
    ap.add_argument("--high-precision", action="store_true", help="Benchmark High-Precision mode (CPU+GPU)")
    ap.add_argument("--model", type=str, default=None, help="ONNX model stem in data/models/ to benchmark")
    args = ap.parse_args()

    rep = run_calibration(
        run_filter=args.run,
        stride=max(1, args.stride),
        synthetic_samples=max(0, args.synthetic),
        high_precision=args.high_precision,
        neural_model_name=args.model,
    )
    print(format_compact_markdown(rep))


if __name__ == "__main__":
    main()

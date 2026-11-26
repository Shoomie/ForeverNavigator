"""LLM-Assisted Model Analysis, Calibration & Surgical Weight-Tuning Pipeline.

Provides deep, token-efficient diagnostics and safe surgical post-training weight
editing for ForeverNavigator's multi-task ONNX minimap models:
  1. Weight & Prototype Anatomy (`inspect_onnx_weights`):
       - Inspects `cls_head` biases, L2 norms, and inter-class prototype cosine
         similarity to identify over-dominant or confusable maps.
  2. Head-to-Head Model Benchmark (`benchmark_models`):
       - Evaluates all (or selected) ONNX models in `data/models/` on:
         (a) Recorded real gameplay logs (`data/dev_gameplay/run_*`) with raw logit
             margin, entropy, cave/leak frame dossiers, heading/centering jitter,
             and full `MinimapTracker` stability metrics.
         (b) Balanced synthetic ground-truth benchmark (Dungeons/Raids at full
             native `data/index/*.npz` resolution + cached Outdoor zones) with
             zone accuracy, `uv_err`, `mm_c`, `hdg_err`, ECE, optimal temperature
             `T*`, and top confusion pairs.
       - Outputs compact `data/calibration/model_analysis.md` & `.json`.
  3. Surgical ONNX Weight Patcher (`apply_surgical_patch`):
       - Directly modifies ONNX `cls_head` / `reg_head` initializers (logit
         temperature scaling, per-class bias nudges, regression bias offsets)
         into a new versioned `<out_stem>.onnx` + `_meta.json` with zero runtime
         overhead and automatic before/after comparison.

Usage:
    python -m src.navigator.model_tuning_pipeline --compare
    python -m src.navigator.model_tuning_pipeline --models forever_minimap_fast forever_minimap_v2_fast --syn-per-map 4
    python -m src.navigator.model_tuning_pipeline --patch-json patch.json --base forever_minimap_v2_fast --out forever_minimap_v2_fast_tuned
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .calibration_pipeline import (
    build_tracker_from_workspace,
    discover_gameplay_runs,
    replay_gameplay_run,
)
from .neural_localizer import (
    NeuralMinimapLocalizer,
    list_available_onnx_models,
)


def _softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / max(1e-4, float(temp))
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)


def _fit_optimal_temperature(logits_mat: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """Grid-search optimal scalar temperature T* minimizing NLL on (N, C) logits.

    Returns `(best_T, ece_before, ece_after)`.
    """
    if len(labels) == 0:
        return 1.0, 0.0, 0.0

    def _nll(t: float) -> float:
        p = _softmax(logits_mat, t)
        idx = np.arange(len(labels))
        return float(-np.mean(np.log(np.clip(p[idx, labels], 1e-9, 1.0))))

    def _ece(t: float, n_bins: int = 10) -> float:
        p = _softmax(logits_mat, t)
        conf = np.max(p, axis=-1)
        pred = np.argmax(p, axis=-1)
        acc = (pred == labels).astype(np.float64)
        ece = 0.0
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
            if np.any(m):
                ece += float(np.mean(m)) * abs(float(np.mean(acc[m])) - float(np.mean(conf[m])))
        return float(ece)

    grid = np.concatenate([np.linspace(0.4, 2.5, 43), np.linspace(2.6, 5.0, 13)])
    best_t = float(min(grid, key=_nll))
    return round(best_t, 3), round(_ece(1.0), 4), round(_ece(best_t), 4)


def inspect_onnx_weights(onnx_path: Path, meta_path: Path) -> Dict[str, Any]:
    """Inspect classifier & regression head weights inside an ONNX model."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    id_to_slug = {int(v): str(k) for k, v in meta.get("class_map", {}).items()}
    num_classes = len(id_to_slug)

    try:
        import onnx
        from onnx import numpy_helper
    except Exception:
        return {"available": False, "reason": "onnx package not installed"}

    model = onnx.load(str(onnx_path))
    inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    cls_w_name, cls_b_name = _find_head_tensors(inits, out_dim=num_classes, prefer_substr="cls_head")
    reg_w_name, reg_b_name = _find_head_tensors(inits, out_dim=7, prefer_substr="reg_head")

    out: Dict[str, Any] = {
        "available": True,
        "num_initializers": len(inits),
        "cls_head": {"weight_name": cls_w_name, "bias_name": cls_b_name},
        "reg_head": {"weight_name": reg_w_name, "bias_name": reg_b_name},
    }

    if cls_w_name and cls_b_name:
        w = inits[cls_w_name].astype(np.float64)
        b = inits[cls_b_name].astype(np.float64)
        if w.shape[0] != num_classes and w.shape[-1] == num_classes:
            w = w.T
        norms = np.linalg.norm(w, axis=-1)
        top_bias_idx = np.argsort(b)[::-1][:6]
        bot_bias_idx = np.argsort(b)[:6]

        # Cosine similarity between class prototypes (excluding self)
        wn = w / (norms[:, None] + 1e-12)
        sim = wn @ wn.T
        np.fill_diagonal(sim, -2.0)
        pairs: List[Tuple[float, str, str]] = []
        for i in range(num_classes):
            j = int(np.argmax(sim[i]))
            if i < j or int(np.argmax(sim[j])) != i:
                pairs.append((float(sim[i, j]), id_to_slug.get(i, str(i)), id_to_slug.get(j, str(j))))
        # Deduplicate unordered pairs
        seen = set()
        top_pairs = []
        for sc, a, c in sorted(pairs, key=lambda x: x[0], reverse=True):
            key = tuple(sorted((a, c)))
            if key in seen:
                continue
            seen.add(key)
            top_pairs.append({"a": a, "b": c, "cosine": round(sc, 4)})
            if len(top_pairs) >= 10:
                break

        out["cls_head"].update({
            "shape": list(w.shape),
            "bias_min": round(float(b.min()), 4),
            "bias_max": round(float(b.max()), 4),
            "bias_std": round(float(b.std()), 4),
            "weight_norm_mean": round(float(norms.mean()), 4),
            "weight_norm_std": round(float(norms.std()), 4),
            "highest_bias_classes": [
                {"slug": id_to_slug.get(int(i), str(i)), "bias": round(float(b[int(i)]), 4), "norm": round(float(norms[int(i)]), 4)}
                for i in top_bias_idx
            ],
            "lowest_bias_classes": [
                {"slug": id_to_slug.get(int(i), str(i)), "bias": round(float(b[int(i)]), 4), "norm": round(float(norms[int(i)]), 4)}
                for i in bot_bias_idx
            ],
            "most_similar_class_pairs": top_pairs,
        })

    if reg_b_name:
        rb = inits[reg_b_name].astype(np.float64)
        out["reg_head"]["bias"] = [round(float(v), 4) for v in rb.tolist()]
    return out


def _find_head_tensors(
    inits: Dict[str, np.ndarray],
    out_dim: int,
    prefer_substr: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Locate (weight_name, bias_name) for a linear head with output dimension `out_dim`."""
    w_cands: List[str] = []
    b_cands: List[str] = []
    for name, arr in inits.items():
        if arr.ndim == 2 and (arr.shape[0] == out_dim or arr.shape[1] == out_dim):
            w_cands.append(name)
        elif arr.ndim == 1 and arr.shape[0] == out_dim:
            b_cands.append(name)

    def _pick(cands: List[str]) -> Optional[str]:
        if not cands:
            return None
        for c in cands:
            if prefer_substr in c:
                return c
        return cands[-1]

    return _pick(w_cands), _pick(b_cands)


def apply_surgical_patch(
    base_stem: str,
    out_stem: str,
    patch: Dict[str, Any],
    models_dir: Path = Path("data/models"),
) -> Tuple[Path, Path]:
    """Apply surgical head edits (temperature, per-class bias deltas, reg bias deltas) to an ONNX model.

    Patch schema (all keys optional):
      {
        "temperature": 1.25,
        "class_bias_deltas": {"sunken_temple": -0.5, "searing_gorge": 0.2},
        "reg_bias_deltas": {"u": 0.0, "v": 0.0, "mm_cx": 0.0, "mm_cy": 0.0, "mm_r": 0.0, "sin": 0.0, "cos": 0.0},
        "note": "Calibrated logit temperature + suppressed cave false positives"
      }
    """
    import onnx
    from onnx import numpy_helper

    src_onnx = models_dir / f"{base_stem}.onnx"
    src_meta = models_dir / f"{base_stem}_meta.json"
    if not src_onnx.exists() or not src_meta.exists():
        raise FileNotFoundError(f"Base model not found: {src_onnx}")

    meta = json.loads(src_meta.read_text(encoding="utf-8"))
    slug_to_id = {str(k): int(v) for k, v in meta.get("class_map", {}).items()}
    num_classes = len(slug_to_id)

    model = onnx.load(str(src_onnx))
    init_by_name = {init.name: init for init in model.graph.initializer}
    arrays = {k: numpy_helper.to_array(v).copy() for k, v in init_by_name.items()}

    cls_w_name, cls_b_name = _find_head_tensors(arrays, out_dim=num_classes, prefer_substr="cls_head")
    _reg_w_name, reg_b_name = _find_head_tensors(arrays, out_dim=7, prefer_substr="reg_head")

    applied: Dict[str, Any] = {"base_stem": base_stem, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    temp = patch.get("temperature")
    if temp is not None and float(temp) > 0 and cls_w_name and cls_b_name:
        inv_t = np.float32(1.0 / float(temp))
        arrays[cls_w_name] = (arrays[cls_w_name].astype(np.float32) * inv_t).astype(arrays[cls_w_name].dtype)
        arrays[cls_b_name] = (arrays[cls_b_name].astype(np.float32) * inv_t).astype(arrays[cls_b_name].dtype)
        applied["temperature"] = float(temp)

    bias_deltas = patch.get("class_bias_deltas") or {}
    if bias_deltas and cls_b_name:
        b = arrays[cls_b_name].astype(np.float32)
        edited = {}
        for slug, delta in bias_deltas.items():
            if slug in slug_to_id:
                cid = slug_to_id[slug]
                b[cid] += float(delta)
                edited[slug] = float(delta)
        arrays[cls_b_name] = b.astype(arrays[cls_b_name].dtype)
        applied["class_bias_deltas"] = edited

    reg_deltas = patch.get("reg_bias_deltas") or {}
    reg_order = ["u", "v", "mm_cx", "mm_cy", "mm_r", "sin", "cos"]
    if reg_deltas and reg_b_name:
        rb = arrays[reg_b_name].astype(np.float32)
        edited_r = {}
        for idx, key in enumerate(reg_order):
            if key in reg_deltas:
                rb[idx] += float(reg_deltas[key])
                edited_r[key] = float(reg_deltas[key])
        arrays[reg_b_name] = rb.astype(arrays[reg_b_name].dtype)
        applied["reg_bias_deltas"] = edited_r

    if patch.get("note"):
        applied["note"] = str(patch["note"])

    # Write modified initializers back into the ONNX graph
    for name in (cls_w_name, cls_b_name, reg_b_name):
        if name and name in init_by_name:
            new_tensor = numpy_helper.from_array(arrays[name], name=name)
            init_by_name[name].CopyFrom(new_tensor)

    dst_onnx = models_dir / f"{out_stem}.onnx"
    dst_meta = models_dir / f"{out_stem}_meta.json"
    onnx.save(model, str(dst_onnx))

    hist = list(meta.get("tuning_history", []))
    hist.append(applied)
    meta["tuning_history"] = hist
    dst_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dst_onnx, dst_meta


def _sample_synthetic_eval_set(
    index_dir: Path = Path("data/index"),
    cache_dir: Path = Path("data/cache/maps"),
    samples_per_map: int = 4,
    seed: int = 1337,
) -> List[Dict[str, Any]]:
    """Build a deterministic, realistic synthetic eval set across Dungeons/Raids (1024px) & cached Outdoor zones."""
    from .world_maps import WorldMapRegistry

    rng = np.random.default_rng(seed)
    reg = WorldMapRegistry(index_dir, cache_dir)
    entries = reg.list_all_maps()

    eval_items: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.category == "Continents":
            continue
        is_dr = entry.category in ("Dungeons", "Raids")
        img: Optional[np.ndarray] = None
        if is_dr:
            npz_p = index_dir / f"{entry.slug}.npz"
            if npz_p.exists():
                try:
                    z = np.load(npz_p, allow_pickle=False)
                    if "map_image" in z:
                        img = np.asarray(z["map_image"])
                except Exception:
                    img = None
        elif entry.cached:
            try:
                d = reg.load_map(entry.slug, preview_only=True)
                img = d.map_image
            except Exception:
                img = None

        if img is None or img.ndim != 3 or img.shape[0] < 256:
            continue
        h, w = img.shape[:2]
        lum = img.mean(axis=-1)
        valid = lum > 25.0
        half_px = max(24, int(round(h * (0.09 if is_dr else 0.065))))
        if half_px * 2 >= min(h, w) - 8:
            continue
        valid[:half_px, :] = False
        valid[-half_px:, :] = False
        valid[:, :half_px] = False
        valid[:, -half_px:] = False
        ys, xs = np.nonzero(valid)
        if len(xs) < 10:
            continue

        for _ in range(samples_per_map):
            pick = int(rng.integers(0, len(xs)))
            cx, cy = int(xs[pick]), int(ys[pick])
            # Realistic off-center ROI framing + arrow heading + slight gain shift
            zoom_f = float(rng.uniform(0.85, 1.25))
            r_px = max(18, int(round(half_px * zoom_f)))
            if cy - r_px < 0 or cy + r_px >= h or cx - r_px < 0 or cx + r_px >= w:
                r_px = half_px
            raw_mm = img[cy - r_px : cy + r_px, cx - r_px : cx + r_px].copy()
            mm_sz = int(rng.integers(140, 196))
            raw_mm = cv2.resize(raw_mm, (mm_sz, mm_sz), interpolation=cv2.INTER_AREA)

            # Circular minimap mask + player arrow at random angle
            angle_deg = float(rng.uniform(0.0, 360.0))
            ang_rad = math.radians(angle_deg)
            canvas = np.full((224, 224, 3), int(rng.integers(12, 34)), dtype=np.uint8)
            mm_cx_px = int(rng.integers(92, 132))
            mm_cy_px = int(rng.integers(92, 132))
            rad = mm_sz // 2
            x0, y0 = mm_cx_px - rad, mm_cy_px - rad
            x1, y1 = x0 + mm_sz, y0 + mm_sz
            if x0 >= 0 and y0 >= 0 and x1 <= 224 and y1 <= 224:
                yy, xx = np.ogrid[:mm_sz, :mm_sz]
                cmask = (xx - rad) ** 2 + (yy - rad) ** 2 <= (rad - 1) ** 2
                roi_dst = canvas[y0:y1, x0:x1]
                roi_dst[cmask] = raw_mm[cmask]
                cv2.circle(canvas, (mm_cx_px, mm_cy_px), rad, (125, 110, 82), 2)
                # Draw directional golden arrow
                tip = (int(round(mm_cx_px + 7 * math.sin(ang_rad))), int(round(mm_cy_px - 7 * math.cos(ang_rad))))
                left = (int(round(mm_cx_px + 5 * math.sin(ang_rad + 2.4))), int(round(mm_cy_px - 5 * math.cos(ang_rad + 2.4))))
                right = (int(round(mm_cx_px + 5 * math.sin(ang_rad - 2.4))), int(round(mm_cy_px - 5 * math.cos(ang_rad - 2.4))))
                cv2.fillConvexPoly(canvas, np.array([tip, left, right], dtype=np.int32), (245, 220, 85))
                gain = float(rng.uniform(0.92, 1.08))
                canvas = np.clip(canvas.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                eval_items.append({
                    "slug": entry.slug,
                    "is_dr": is_dr,
                    "img": canvas,
                    "u": cx / float(w),
                    "v": cy / float(h),
                    "mm_cx": mm_cx_px / 224.0,
                    "mm_cy": mm_cy_px / 224.0,
                    "mm_r": rad / 224.0,
                    "heading_deg": angle_deg,
                })
    return eval_items


def evaluate_model_raw_on_gameplay(
    nn: NeuralMinimapLocalizer,
    runs: Sequence[Path],
    stride: int = 4,
) -> Dict[str, Any]:
    """Run raw neural diagnostics (latency, margin, entropy, leaks, jitter) on real gameplay runs."""
    run_summaries: List[Dict[str, Any]] = []
    all_ms: List[float] = []

    for run_dir in runs:
        mm_files = sorted((run_dir / "minimap").glob("*.png"))[:: max(1, stride)]
        if not mm_files:
            continue
        slug_counts: Dict[str, int] = {}
        probs_top1: List[float] = []
        margins: List[float] = []
        entropies: List[float] = []
        hdgs: List[float] = []
        mm_cs: List[Tuple[float, float]] = []
        frame_records: List[Dict[str, Any]] = []

        for fpath in mm_files:
            bgr = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if float((rgb.mean(axis=-1) > 18.0).mean()) < 0.05:
                continue
            raw = nn.predict_raw(rgb)
            if raw is None:
                continue
            logits, reg, _emb, ms = raw
            all_ms.append(ms)
            probs = _softmax(logits, 1.0)
            order = np.argsort(probs)[::-1]
            top1_i, top2_i = int(order[0]), int(order[1])
            slug1 = nn.id_to_slug.get(top1_i, f"cls_{top1_i}")
            slug2 = nn.id_to_slug.get(top2_i, f"cls_{top2_i}")
            p1 = float(probs[top1_i])
            margin = float(logits[top1_i] - logits[top2_i])
            ent = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
            hdg = float(math.degrees(math.atan2(float(reg[5]), float(reg[6]))) % 360.0)

            slug_counts[slug1] = slug_counts.get(slug1, 0) + 1
            probs_top1.append(p1)
            margins.append(margin)
            entropies.append(ent)
            hdgs.append(hdg)
            mm_cs.append((float(reg[2]), float(reg[3])))
            frame_records.append({
                "file": f"{run_dir.name}/minimap/{fpath.name}",
                "top1": slug1,
                "p1": round(p1, 3),
                "top2": slug2,
                "p2": round(float(probs[top2_i]), 3),
                "margin": round(margin, 2),
                "entropy": round(ent, 2),
            })

        if not probs_top1:
            continue
        dom_sorted = sorted(slug_counts.items(), key=lambda kv: kv[1], reverse=True)
        dom_slug = dom_sorted[0][0]
        dom_share = dom_sorted[0][1] / float(len(probs_top1))
        # Find leak / disagreement frames (where raw top1 != dominant zone)
        leaks = [r for r in frame_records if r["top1"] != dom_slug]
        leaks.sort(key=lambda r: r["p1"], reverse=True)

        # Circular frame-to-frame heading delta
        hdg_deltas = []
        for a, b in zip(hdgs, hdgs[1:]):
            d = abs(((b - a + 180.0) % 360.0) - 180.0)
            hdg_deltas.append(d)
        mm_arr = np.array(mm_cs, dtype=np.float64)

        run_summaries.append({
            "run_name": run_dir.name,
            "frames": len(probs_top1),
            "dominant_slug": dom_slug,
            "dominant_raw_share": round(dom_share, 4),
            "top_raw_slugs": dom_sorted[:5],
            "mean_prob": round(float(np.mean(probs_top1)), 4),
            "p10_prob": round(float(np.percentile(probs_top1, 10)), 4),
            "mean_margin": round(float(np.mean(margins)), 3),
            "mean_entropy": round(float(np.mean(entropies)), 3),
            "median_hdg_step_deg": round(float(np.median(hdg_deltas)), 2) if hdg_deltas else 0.0,
            "mm_center_std": round(float(np.mean(np.std(mm_arr, axis=0))), 4),
            "top_raw_leaks": leaks[:5],
        })

    ms_arr = np.array(all_ms, dtype=np.float64) if all_ms else np.zeros(1)
    return {
        "raw_p50_ms": round(float(np.median(ms_arr)), 2),
        "raw_p95_ms": round(float(np.percentile(ms_arr, 95)), 2),
        "runs": run_summaries,
    }


def evaluate_model_on_synthetic(
    nn: NeuralMinimapLocalizer,
    eval_items: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate raw ONNX model on balanced synthetic ground-truth crops (calibration + accuracy + confusion)."""
    logits_list: List[np.ndarray] = []
    labels_list: List[int] = []
    dr_hits = dr_n = out_hits = out_n = 0
    dr_uv: List[float] = []
    out_uv: List[float] = []
    mm_errs: List[float] = []
    hdg_errs: List[float] = []
    confusions: Dict[Tuple[str, str], int] = {}

    for item in eval_items:
        cid = nn.slug_to_id.get(item["slug"])
        if cid is None:
            continue
        raw = nn.predict_raw(item["img"])
        if raw is None:
            continue
        logits, reg, _emb, _ms = raw
        logits_list.append(logits)
        labels_list.append(cid)

        pred_id = int(np.argmax(logits))
        pred_slug = nn.id_to_slug.get(pred_id, f"cls_{pred_id}")
        hit = (pred_id == cid)
        uv_err = float(math.hypot(float(reg[0]) - item["u"], float(reg[1]) - item["v"]))
        mm_err = float(math.hypot(float(reg[2]) - item["mm_cx"], float(reg[3]) - item["mm_cy"]))
        pred_hdg = float(math.degrees(math.atan2(float(reg[5]), float(reg[6]))) % 360.0)
        hdg_err = abs(((pred_hdg - item["heading_deg"] + 180.0) % 360.0) - 180.0)

        mm_errs.append(mm_err)
        hdg_errs.append(hdg_err)
        if item["is_dr"]:
            dr_n += 1
            if hit:
                dr_hits += 1
            dr_uv.append(uv_err)
        else:
            out_n += 1
            if hit:
                out_hits += 1
            out_uv.append(uv_err)

        if not hit:
            key = (item["slug"], pred_slug)
            confusions[key] = confusions.get(key, 0) + 1

    if not labels_list:
        return {"samples": 0}

    l_mat = np.stack(logits_list, axis=0)
    y_vec = np.asarray(labels_list, dtype=np.int64)
    opt_t, ece_before, ece_after = _fit_optimal_temperature(l_mat, y_vec)
    top_conf = [
        {"true": k[0], "pred": k[1], "count": v}
        for k, v in sorted(confusions.items(), key=lambda kv: kv[1], reverse=True)[:10]
    ]

    total = dr_n + out_n
    return {
        "samples": total,
        "dr_samples": dr_n,
        "outdoor_samples": out_n,
        "dr_acc": round(dr_hits / max(1, dr_n), 4),
        "outdoor_acc": round(out_hits / max(1, out_n), 4),
        "all_acc": round((dr_hits + out_hits) / max(1, total), 4),
        "dr_uv_err": round(float(np.mean(dr_uv)), 4) if dr_uv else 0.0,
        "outdoor_uv_err": round(float(np.mean(out_uv)), 4) if out_uv else 0.0,
        "mm_c_err": round(float(np.mean(mm_errs)), 4),
        "hdg_err_deg": round(float(np.mean(hdg_errs)), 2),
        "optimal_temp": opt_t,
        "ece_before": ece_before,
        "ece_after_temp": ece_after,
        "top_confusions": top_conf,
    }


def benchmark_models(
    model_stems: Optional[Sequence[str]] = None,
    models_dir: Path = Path("data/models"),
    dev_root: Path = Path("data/dev_gameplay"),
    out_dir: Path = Path("data/calibration"),
    gameplay_stride: int = 4,
    syn_per_map: int = 4,
    include_tracker_replay: bool = True,
) -> Dict[str, Any]:
    """Run full analytical comparison across selected (or all) ONNX models."""
    t0 = time.perf_counter()
    avail = list_available_onnx_models(models_dir)
    if model_stems:
        stem_set = set(model_stems)
        selected = [m for m in avail if m[0] in stem_set]
    else:
        selected = avail

    runs = discover_gameplay_runs(dev_root)
    eval_items = _sample_synthetic_eval_set(samples_per_map=syn_per_map) if syn_per_map > 0 else []

    results: Dict[str, Any] = {}
    for stem, onnx_p, meta_p in selected:
        nn = NeuralMinimapLocalizer(model_path=onnx_p, meta_path=meta_p)
        entry: Dict[str, Any] = {
            "stem": stem,
            "size_mb": round(onnx_p.stat().st_size / (1024 * 1024), 2),
            "weights_anatomy": inspect_onnx_weights(onnx_p, meta_p),
            "raw_gameplay": evaluate_model_raw_on_gameplay(nn, runs, stride=gameplay_stride),
            "synthetic_eval": evaluate_model_on_synthetic(nn, eval_items),
        }
        if include_tracker_replay and runs:
            tracker, _reg = build_tracker_from_workspace(high_precision=False)
            tracker.set_neural_model(stem, models_dir=models_dir)
            entry["tracker_replay"] = [
                {
                    "run_name": m.run_name,
                    "frames": m.frames,
                    "mean_conf": m.mean_conf,
                    "zone_switches": m.zone_switches,
                    "ping_pong": m.ping_pong_switches,
                    "false_dungeon_jumps": m.false_dungeon_jumps,
                    "mean_ms": m.mean_ms,
                    "p95_ms": m.p95_ms,
                    "dominant_zones": m.dominant_zones[:3],
                }
                for m in (replay_gameplay_run(r, tracker, stride=gameplay_stride) for r in runs)
            ]
            tracker.close()
        results[stem] = entry

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "gameplay_stride": gameplay_stride,
        "syn_samples": len(eval_items),
        "models": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = format_analysis_markdown(report)
    (out_dir / "model_analysis.md").write_text(md, encoding="utf-8")
    return report


def format_analysis_markdown(report: Dict[str, Any]) -> str:
    """Render a concise, LLM-optimized diagnostic dossier for manual model tuning."""
    lines = [
        "# ForeverNavigator — LLM Model Analysis & Tuning Dossier",
        f"- Generated: `{report['generated_at']}` | Runtime: `{report['elapsed_sec']:.1f}s` | Synthetic eval crops: `{report['syn_samples']}`",
        "",
        "## 1. Synthetic Ground-Truth & Calibration Comparison",
        "| Model | Size | D/R Acc | Outdoor Acc | All Acc | D/R `uv_err` | `mm_c` | `hdg` | ECE (1.0 -> T*) | Optimal `T*` | Raw CPU p50/p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stem, m in report.get("models", {}).items():
        s = m.get("synthetic_eval", {})
        g = m.get("raw_gameplay", {})
        lines.append(
            f"| `{stem}` | {m['size_mb']:.1f} MB | {s.get('dr_acc', 0)*100:.1f}% | {s.get('outdoor_acc', 0)*100:.1f}% | "
            f"{s.get('all_acc', 0)*100:.1f}% | {s.get('dr_uv_err', 0):.4f} | {s.get('mm_c_err', 0):.4f} | "
            f"{s.get('hdg_err_deg', 0):.1f}° | {s.get('ece_before', 0):.3f} → {s.get('ece_after_temp', 0):.3f} | "
            f"**{s.get('optimal_temp', 1.0):.2f}** | {g.get('raw_p50_ms', 0):.1f}/{g.get('raw_p95_ms', 0):.1f} ms |"
        )

    lines.extend([
        "",
        "## 2. Real Gameplay Holdout Logs (Raw Neural + Full Tracker)",
        "| Model | Run | Raw Dom Share | Mean Prob | Margin | Entropy | Tracker Switches | Ping-Pong | False Dng | Tracker ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for stem, m in report.get("models", {}).items():
        tr_by_run = {r["run_name"]: r for r in m.get("tracker_replay", [])}
        for r in m.get("raw_gameplay", {}).get("runs", []):
            tr = tr_by_run.get(r["run_name"], {})
            lines.append(
                f"| `{stem}` | `{r['run_name']}` ({r['dominant_slug']}) | {r['dominant_raw_share']*100:.1f}% | "
                f"{r['mean_prob']:.2f} | {r['mean_margin']:.2f} | {r['mean_entropy']:.2f} | "
                f"{tr.get('zone_switches', '-')} | {tr.get('ping_pong', '-')} | {tr.get('false_dungeon_jumps', '-')} | "
                f"{tr.get('mean_ms', '-')} |"
            )

    lines.extend(["", "## 3. Weight Anatomy & Surgical Tuning Signals"])
    for stem, m in report.get("models", {}).items():
        w = m.get("weights_anatomy", {})
        s = m.get("synthetic_eval", {})
        lines.append(f"### `{stem}`")
        if w.get("available") and "highest_bias_classes" in w.get("cls_head", {}):
            ch = w["cls_head"]
            hi = ", ".join(f"{x['slug']}({x['bias']:+.2f})" for x in ch["highest_bias_classes"][:4])
            sim = ", ".join(f"{x['a']}↔{x['b']}({x['cosine']:.2f})" for x in ch.get("most_similar_class_pairs", [])[:4])
            lines.append(f"- `cls_head` bias range: `[{ch['bias_min']:+.3f}, {ch['bias_max']:+.3f}]` | Highest bias: {hi}")
            lines.append(f"- Closest prototype pairs: {sim}")
        if s.get("top_confusions"):
            cf = ", ".join(f"{x['true']}→{x['pred']}({x['count']})" for x in s["top_confusions"][:5])
            lines.append(f"- Top synthetic confusions: {cf}")
        for r in m.get("raw_gameplay", {}).get("runs", []):
            if r.get("top_raw_leaks"):
                lk = ", ".join(f"{x['top1']}({x['p1']:.2f}, m={x['margin']:.1f})" for x in r["top_raw_leaks"][:3])
                lines.append(f"- Raw leaks on `{r['run_name']}`: {lk}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="ForeverNavigator LLM Model Analysis & Surgical Tuning Pipeline")
    ap.add_argument("--compare", action="store_true", help="Compare all available ONNX models in data/models/")
    ap.add_argument("--models", nargs="+", default=None, help="Specific model stems to benchmark")
    ap.add_argument("--stride", type=int, default=4, help="Frame stride across real gameplay logs")
    ap.add_argument("--syn-per-map", type=int, default=4, help="Synthetic eval crops per map")
    ap.add_argument("--no-tracker", action="store_true", help="Skip full MinimapTracker replay (raw neural only)")
    ap.add_argument("--patch-json", type=str, default=None, help="Path to surgical JSON patch to apply to --base")
    ap.add_argument("--base", type=str, default="forever_minimap_v2_fast", help="Base model stem for --patch-json")
    ap.add_argument("--out", type=str, default=None, help="Output model stem for --patch-json")
    args = ap.parse_args()

    if args.patch_json:
        patch = json.loads(Path(args.patch_json).read_text(encoding="utf-8"))
        out_stem = args.out or f"{args.base}_tuned"
        dst_onnx, dst_meta = apply_surgical_patch(args.base, out_stem, patch)
        print(f"Patched model saved: {dst_onnx} + {dst_meta}")
        rep = benchmark_models(
            model_stems=[args.base, out_stem],
            gameplay_stride=max(1, args.stride),
            syn_per_map=max(0, args.syn_per_map),
            include_tracker_replay=not args.no_tracker,
        )
        print(format_analysis_markdown(rep))
        return

    rep = benchmark_models(
        model_stems=args.models,
        gameplay_stride=max(1, args.stride),
        syn_per_map=max(0, args.syn_per_map),
        include_tracker_replay=not args.no_tracker,
    )
    print(format_analysis_markdown(rep))


if __name__ == "__main__":
    main()

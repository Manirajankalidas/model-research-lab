"""Builds comparison.json / summary.csv for an inference or comparison job from the per-model results.

Agreement: without ground-truth labels we can still check how closely a converted model reproduces the
baseline model's boxes (same class, IoU >= 0.5) on every frame. F1 = 1.0 means identical detections.
"""
import csv
import json
from pathlib import Path

MAX_POINTS = 1500  # chart series are averaged down to this many points


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(ref, pred, thr=0.5):
    """Greedy same-class matching, highest confidence first. Returns (tp, fp, fn, sum of matched IoU)."""
    used, tp, iou_sum = set(), 0, 0.0
    for p in sorted(pred, key=lambda b: -b[1]):
        best, best_j = thr, None
        for j, r in enumerate(ref):
            if j in used or r[0] != p[0]:
                continue
            v = iou(p[2:], r[2:])
            if v >= best:
                best, best_j = v, j
        if best_j is not None:
            used.add(best_j)
            tp += 1
            iou_sum += best
    return tp, len(pred) - tp, len(ref) - tp, iou_sum


def read_dets(path, skip):
    frames = {}
    try:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d["f"] >= skip:
                    frames[d["f"]] = d["b"]
    except OSError:
        pass
    return frames


def agreement(base_path, path, skip):
    base, other = read_dets(base_path, skip), read_dets(path, skip)
    common = sorted(set(base) & set(other))
    if not common:
        return None
    tp = fp = fn = 0
    iou_sum, same_count = 0.0, 0
    for f in common:
        t, p, n, s = match(base[f], other[f])
        tp, fp, fn, iou_sum = tp + t, fp + p, fn + n, iou_sum + s
        same_count += len(base[f]) == len(other[f])
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"frames": len(common), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "mean_iou": round(iou_sum / tp, 4) if tp else None, "same_count_pct": round(100 * same_count / len(common), 2),
            "extra_boxes": fp, "missed_boxes": fn}


def downsample(xs, ys, n=MAX_POINTS):
    if len(xs) <= n:
        return xs, ys
    step = len(xs) / n
    ox, oy = [], []
    for i in range(n):
        a, b = int(i * step), max(int((i + 1) * step), int(i * step) + 1)
        seg = [v for v in ys[a:b] if v is not None]
        ox.append(xs[a])
        oy.append(round(sum(seg) / len(seg), 3) if seg else None)
    return ox, oy


def read_series(rdir):
    s = {}
    try:
        with open(rdir / "frames.csv") as f:
            rows = list(csv.DictReader(f))
        x = [int(r["frame"]) for r in rows]
        for key in ("latency_ms", "inference_ms", "detections"):
            s[key] = downsample(x, [float(r[key]) for r in rows])
    except (OSError, KeyError, ValueError):
        pass
    try:
        with open(rdir / "system.csv") as f:
            rows = list(csv.DictReader(f))
        x = [float(r["t"]) for r in rows]
        for key in ("power_w", "gpu_pct", "gpu_temp_c"):
            ys = [float(r[key]) if r.get(key) else None for r in rows]
            if any(v is not None for v in ys):
                s[key] = downsample(x, ys)
    except (OSError, KeyError, ValueError):
        pass
    return s


def g(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def build(job_dir, models, baseline=0, env=None):
    """models: list of dicts from the job (key, label, ...); results are in job_dir/results/<i>/."""
    job_dir = Path(job_dir)
    rows = []
    for i, m in enumerate(models):
        rdir = job_dir / "results" / str(i)
        summ = load(rdir / "summary.json")
        rows.append({"index": i, "key": m["key"], "label": m["label"], "ok": summ is not None,
                     "summary": summ, "val": load(rdir / "val.json"), "series": read_series(rdir) if summ else {}})

    base = rows[baseline] if 0 <= baseline < len(rows) and rows[baseline]["ok"] else None
    for r in rows:
        if not r["ok"] or not base:
            continue
        b_lat, lat = g(base, "summary", "latency_ms", "mean"), g(r, "summary", "latency_ms", "mean")
        r["speedup"] = round(b_lat / lat, 3) if b_lat and lat else None
        if r is not base and len(rows) > 1:
            skip = r["summary"]["settings"].get("warmup", 0) if r["summary"]["source"]["type"] == "video" else 0
            r["agreement"] = agreement(job_dir / "results" / str(base["index"]) / "detections.jsonl",
                                       job_dir / "results" / str(r["index"]) / "detections.jsonl", skip)

    result = {"baseline": baseline, "env": env, "models": rows}
    (job_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    write_csv(job_dir / "summary.csv", rows, base)
    return result


COLUMNS = [
    ("model", lambda r: r["label"]),
    ("format", lambda r: g(r, "summary", "model", "format")),
    ("precision", lambda r: g(r, "summary", "model", "precision")),
    ("size_mb", lambda r: g(r, "summary", "model", "size_mb")),
    ("imgsz", lambda r: g(r, "summary", "settings", "imgsz")),
    ("device", lambda r: g(r, "summary", "runtime", "device")),
    ("measured_frames", lambda r: g(r, "summary", "measured_frames")),
    ("preprocess_ms_mean", lambda r: g(r, "summary", "preprocess_ms", "mean")),
    ("inference_ms_mean", lambda r: g(r, "summary", "inference_ms", "mean")),
    ("inference_ms_std", lambda r: g(r, "summary", "inference_ms", "std")),
    ("inference_ms_p50", lambda r: g(r, "summary", "inference_ms", "p50")),
    ("inference_ms_p95", lambda r: g(r, "summary", "inference_ms", "p95")),
    ("postprocess_ms_mean", lambda r: g(r, "summary", "postprocess_ms", "mean")),
    ("latency_ms_mean", lambda r: g(r, "summary", "latency_ms", "mean")),
    ("latency_ms_p95", lambda r: g(r, "summary", "latency_ms", "p95")),
    ("fps_model", lambda r: g(r, "summary", "fps_model")),
    ("fps_pipeline", lambda r: g(r, "summary", "fps_pipeline")),
    ("speedup_vs_baseline", lambda r: r.get("speedup")),
    ("detections_total", lambda r: g(r, "summary", "detections", "total")),
    ("detections_per_frame", lambda r: g(r, "summary", "detections", "per_frame_mean")),
    ("mean_conf", lambda r: g(r, "summary", "detections", "mean_conf")),
    ("agreement_f1", lambda r: g(r, "agreement", "f1")),
    ("agreement_precision", lambda r: g(r, "agreement", "precision")),
    ("agreement_recall", lambda r: g(r, "agreement", "recall")),
    ("agreement_mean_iou", lambda r: g(r, "agreement", "mean_iou")),
    ("power_w_mean", lambda r: g(r, "summary", "system", "power_w_mean")),
    ("fps_per_watt", lambda r: g(r, "summary", "fps_per_watt")),
    ("gpu_util_mean", lambda r: g(r, "summary", "system", "gpu_util_mean")),
    ("gpu_temp_max", lambda r: g(r, "summary", "system", "gpu_temp_max")),
    ("ram_mb_max", lambda r: g(r, "summary", "system", "ram_mb_max")),
    ("map50", lambda r: g(r, "val", "map50")),
    ("map50_95", lambda r: g(r, "val", "map50_95")),
    ("val_precision", lambda r: g(r, "val", "precision")),
    ("val_recall", lambda r: g(r, "val", "recall")),
]


def write_csv(path, rows, base):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in COLUMNS] + ["baseline"])
        for r in rows:
            w.writerow([fn(r) if r["ok"] else (r["label"] if c == "model" else "") for c, fn in COLUMNS]
                       + [int(r is base)])

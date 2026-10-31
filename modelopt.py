#!/usr/bin/env python3
"""Tiny model conversion + comparison tool for YOLO on Jetson.

  python modelopt.py convert --weights yolov8n.pt --data coco8.yaml
  python modelopt.py bench   --data coco8.yaml
  python modelopt.py report
  python modelopt.py all     --weights yolov8n.pt --data coco8.yaml

TensorRT engines are device specific: run `convert` on the Jetson itself.
"""
import argparse
import html
import json
import shutil
import statistics as st
import time
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import ASSETS

# variant name -> export kwargs (None = copy the original weights)
VARIANTS = {
    "pt_fp32": None,
    "onnx_fp32": dict(format="onnx", simplify=True),
    "trt_fp32": dict(format="engine"),
    "trt_fp16": dict(format="engine", half=True),
    "trt_int8": dict(format="engine", int8=True),
}


def device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def calib_yaml(weights, calib_dir, out):
    """Build a data yaml from a plain folder of images (no labels needed) for INT8 calibration."""
    import yaml
    imgs = [p for p in calib_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not imgs:
        raise SystemExit(f"no images found in {calib_dir}")
    root = out / "calib"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(imgs):  # empty label file = background image, which is fine for calibration
        name = f"{i:05d}{p.suffix.lower()}"
        shutil.copy(p, root / "images" / name)
        (root / "labels" / f"{i:05d}.txt").write_text("")
    names = YOLO(str(weights)).names
    cfg = root / "data.yaml"
    cfg.write_text(yaml.safe_dump({"path": str(root.resolve()), "train": "images", "val": "images", "names": names}))
    print(f"[calib] {len(imgs)} images -> {cfg}")
    return str(cfg)


def convert(a):
    out = Path(a.out)
    models, work = out / "models", out / "work"
    models.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    src = work / Path(a.weights).name
    if not src.exists():
        YOLO(a.weights)  # triggers download if it is a known name like yolov8n.pt
        found = Path(a.weights)
        shutil.copy(found if found.exists() else Path.cwd() / found.name, src)

    if a.calib_dir and "trt_int8" in a.variants:
        a.data = calib_yaml(src, Path(a.calib_dir), out)

    for name in a.variants:
        kw = VARIANTS[name]
        dst = models / (name + (".pt" if kw is None else ".onnx" if kw["format"] == "onnx" else ".engine"))
        if dst.exists() and not a.force:
            print(f"[skip] {dst.name} exists")
            continue
        print(f"[convert] {name}")
        try:
            if kw is None:
                shutil.copy(src, dst)
                continue
            kw = dict(kw, imgsz=a.imgsz, batch=1, device=device())
            if kw.get("int8"):
                kw["data"] = a.data  # calibration images come from this dataset
            t = time.time()
            produced = Path(YOLO(str(src)).export(**kw))
            shutil.move(str(produced), dst)
            print(f"[ok] {dst.name} in {time.time() - t:.0f}s")
        except Exception as e:  # keep going so one failure does not kill the run
            print(f"[FAIL] {name}: {e}")


def bench(a):
    out = Path(a.out)
    files = sorted((out / "models").glob("*.*"), key=lambda p: list(VARIANTS).index(p.stem)
                   if p.stem in VARIANTS else 99)
    img = a.image or str(ASSETS / "bus.jpg")
    results = []
    for f in files:
        print(f"[bench] {f.name}")
        r = {"name": f.stem, "file": f.name, "size_mb": round(f.stat().st_size / 1e6, 1)}
        try:
            m = YOLO(str(f), task="detect")
            kw = dict(imgsz=a.imgsz, device=device(), verbose=False)
            for _ in range(a.warmup):
                m.predict(img, **kw)
            inf, tot = [], []
            for _ in range(a.runs):
                t = time.perf_counter()
                res = m.predict(img, **kw)[0]
                tot.append((time.perf_counter() - t) * 1000)
                inf.append(res.speed["inference"])
            r.update(
                infer_ms_p50=round(st.median(inf), 2),
                infer_ms_p95=round(sorted(inf)[int(len(inf) * 0.95) - 1], 2),
                e2e_ms_p50=round(st.median(tot), 2),
                fps=round(1000 / st.median(tot), 1),
                detections=len(res.boxes),
            )
            if a.data:
                v = m.val(data=a.data, imgsz=a.imgsz, batch=1, device=device(), plots=False, verbose=False)
                r.update(map50=round(float(v.box.map50), 4), map50_95=round(float(v.box.map), 4))
        except Exception as e:
            r["error"] = str(e)[:200]
            print(f"[FAIL] {f.name}: {e}")
        results.append(r)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"saved {out / 'results.json'}")


def report(a):
    out = Path(a.out)
    rs = json.loads((out / "results.json").read_text())
    ok = [r for r in rs if "error" not in r]
    base = next((r for r in ok if r["name"] == "trt_fp32"), ok[0] if ok else None)
    max_fps = max((r["fps"] for r in ok), default=1)

    def delta(r, k):
        if not base or k not in r or k not in base:
            return ""
        return f" <small>({r[k] - base[k]:+.4f})</small>"

    rows = []
    for r in rs:
        if "error" in r:
            rows.append(f"<tr><td>{r['name']}</td><td colspan=7 class=err>{html.escape(r['error'])}</td></tr>")
            continue
        speedup = f"{base['infer_ms_p50'] / r['infer_ms_p50']:.2f}x" if base else ""
        rows.append(
            f"<tr><td>{r['name']}</td><td>{r['size_mb']}</td><td>{r['infer_ms_p50']}</td>"
            f"<td>{r['infer_ms_p95']}</td><td>{speedup}</td>"
            f"<td><div class=bar style='width:{r['fps'] / max_fps * 120:.0f}px'></div> {r['fps']}</td>"
            f"<td>{r.get('map50', '-')}{delta(r, 'map50')}</td>"
            f"<td>{r.get('map50_95', '-')}{delta(r, 'map50_95')}</td></tr>")
    page = f"""<!doctype html><meta charset=utf-8><title>Model comparison</title>
<style>body{{font:14px sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px 10px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.bar{{display:inline-block;height:10px;background:#3b82f6}}.err{{color:#b91c1c;text-align:left}}</style>
<h2>Model comparison</h2><p>Baseline for speedup/deltas: <b>{base['name'] if base else '-'}</b>.
Latency = pure inference (ms, p50/p95); FPS is end-to-end incl. pre/post-processing.</p>
<table><tr><th>variant<th>size MB<th>infer p50 ms<th>infer p95 ms<th>speedup<th>FPS<th>mAP50<th>mAP50-95</tr>
{''.join(rows)}</table>"""
    (out / "report.html").write_text(page)
    print(f"saved {out / 'report.html'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["convert", "bench", "report", "all"])
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--data", default="coco8.yaml", help="dataset yaml: INT8 calibration + mAP")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--calib-dir", help="folder of images for INT8 calibration (alternative to --data)")
    p.add_argument("--image", help="image used for latency timing")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--force", action="store_true", help="re-convert existing files")
    a = p.parse_args()
    if a.cmd in ("convert", "all"):
        convert(a)
    if a.cmd in ("bench", "all"):
        bench(a)
    if a.cmd in ("report", "all"):
        report(a)


if __name__ == "__main__":
    main()

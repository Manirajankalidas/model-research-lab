#!/usr/bin/env python3
"""Run one or more converted models on a video and save annotated outputs.

  python run_video.py --video abc.mp4 --models jobs/<id>/models/trt_fp16.engine jobs/<id>/models/trt_int8.engine
  python run_video.py --video abc.mp4 --models a.engine b.engine --no-save    # speed only
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def run(model_path, video, a):
    model = YOLO(model_path, task="detect")
    cap = cv2.VideoCapture(video)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = None
    out_path = Path(a.out) / f"{Path(model_path).stem}_{Path(video).stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    infer, counts, frames = [], Counter(), 0
    t0 = time.perf_counter()
    for r in model.predict(video, imgsz=a.imgsz, conf=a.conf, stream=True, verbose=False, device=a.device):
        frames += 1
        infer.append(r.speed["inference"])
        counts.update(model.names[int(c)] for c in r.boxes.cls)
        if not a.no_save:
            img = r.plot()
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
            writer.write(img)
        if frames % 200 == 0:
            print(f"  {frames}/{total} frames", flush=True)
    wall = time.perf_counter() - t0
    if writer:
        writer.release()

    infer.sort()
    return {
        "model": Path(model_path).name, "frames": frames,
        "infer_ms_p50": round(infer[len(infer) // 2], 2) if infer else 0,
        "fps_wall": round(frames / wall, 1),
        "detections": sum(counts.values()), "per_class": dict(counts),
        "output": None if a.no_save else str(out_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--models", nargs="+", required=True, help=".pt / .onnx / .engine files")
    p.add_argument("--imgsz", type=int, default=640, help="must match the size used for conversion")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default=0, help="0 for GPU, 'cpu' for CPU")
    p.add_argument("--out", default="video_out")
    p.add_argument("--json", help="also write the results table to this json file")
    p.add_argument("--no-save", action="store_true", help="skip writing video (faster, measures speed only)")
    a = p.parse_args()
    if str(a.device).isdigit():
        a.device = int(a.device)

    rows = []
    for m in a.models:
        print(f"[run] {m}")
        rows.append(run(m, a.video, a))

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))

    print(f"\n{'model':24}{'frames':>8}{'infer p50 ms':>14}{'FPS':>8}{'dets':>8}")
    for r in rows:
        print(f"{r['model']:24}{r['frames']:>8}{r['infer_ms_p50']:>14}{r['fps_wall']:>8}{r['detections']:>8}")
    for r in rows:
        print(f"\n{r['model']} per-class: {r['per_class']}" + (f"\n  video: {r['output']}" if r["output"] else ""))


if __name__ == "__main__":
    main()

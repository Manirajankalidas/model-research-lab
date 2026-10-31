#!/usr/bin/env python3
"""Heavy lifting for the web app. app.py runs each command in its own process, so GPU memory is
released after every step and a crash cannot take the web server down.

  python worker.py env     --out data/env.json
  python worker.py convert --group-dir data/models/<name> --variants onnx_fp32 trt_fp16 --imgsz 640
  python worker.py infer   --model data/models/<name>/trt_fp16.engine --source video.mp4 --out <dir>
  python worker.py val     --model data/models/<name>/trt_fp16.engine --data data.yaml --out <dir>
"""
import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# variant -> (file extension, export kwargs). pt_fp32 is the uploaded weights themselves.
VARIANTS = {
    "pt_fp32": (".pt", None),
    "onnx_fp32": (".onnx", dict(format="onnx", simplify=True)),
    "trt_fp32": (".engine", dict(format="engine")),
    "trt_fp16": (".engine", dict(format="engine", half=True)),
    "trt_int8": (".engine", dict(format="engine", int8=True)),  # import only for now; conversion on hold
}
FORMAT = {"pt": "PyTorch", "onnx": "ONNX", "trt": "TensorRT"}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(path, obj):
    """Atomic write: the web server may read the file while we write it."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def read_text(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def run_out(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pick_device(want, model_path):
    if str(model_path).endswith(".engine"):
        if not cuda_ok():
            raise SystemExit("TensorRT engines need a CUDA GPU; none found on this device")
        return 0
    if want == "cpu":
        return "cpu"
    if want == "gpu" and not cuda_ok():
        raise SystemExit("GPU requested but CUDA is not available")
    return 0 if cuda_ok() else "cpu"


# ----------------------------------------------------------------------------------------- env
def device_model():
    return read_text("/proc/device-tree/model").strip("\x00\n ") or platform.node()


def cmd_env(a):
    info = {"collected": now(), "device": device_model(), "python": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}"}
    rel = read_text("/etc/nv_tegra_release")
    m = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", rel)
    info["l4t"] = f"R{m.group(1)}.{m.group(2)}" if m else None
    info["jetpack"] = run_out(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"]) or None
    pm = run_out(["nvpmodel", "-q"])
    info["power_mode"] = pm.splitlines()[0].replace("NV Power Mode:", "").strip() if pm else None
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if info["cuda"] else None
    except Exception as e:
        info["torch"], info["cuda"] = f"missing ({type(e).__name__})", False
    for mod in ("tensorrt", "ultralytics", "onnxruntime", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import onnxruntime
        info["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except Exception:
        info["onnxruntime_providers"] = []
    info["ffmpeg_h264"] = h264_available()
    info["tegrastats"] = bool(shutil.which("tegrastats"))
    write_json(a.out, info)
    print(json.dumps(info, indent=2))


# ------------------------------------------------------------------------------------- convert
def model_info(pt):
    """Basic facts about a .pt model, stored once per model group."""
    from ultralytics import YOLO
    m = YOLO(str(pt))
    info = {"task": m.task, "names": m.names, "num_classes": len(m.names)}
    train = (getattr(m, "ckpt", None) or {}).get("train_args", {}) or {}
    # custom models keep the base model they were trained from in train_args (e.g. yolov8n.pt / yolo11s.pt)
    arch = (m.model.yaml or {}).get("yaml_file") or train.get("model") or ""
    info["arch"] = Path(str(arch)).stem or None
    info["train_imgsz"] = train.get("imgsz")
    info["train_epochs"] = train.get("epochs")
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        info["params_m"] = round(get_num_params(m.model) / 1e6, 2)
        info["gflops"] = round(get_flops(m.model, train.get("imgsz") or 640), 1)
    except Exception:
        pass
    return info


def variant_meta(variant, path, imgsz, **extra):
    kind, precision = variant.split("_")
    return dict(variant=variant, format=FORMAT[kind], precision=precision.upper(), file=Path(path).name,
                imgsz=imgsz, size_mb=round(Path(path).stat().st_size / 1e6, 2), created=now(), **extra)


def cmd_convert(a):
    gdir = Path(a.group_dir)
    pt = gdir / "pt_fp32.pt"
    prog_path = gdir / "progress.json"
    prog = {"total": len(a.variants), "step": 0, "current": "reading model", "done": [], "failed": {}}
    write_json(prog_path, prog)

    group = json.loads(read_text(gdir / "model.json") or "{}")
    if not group.get("from_pt"):  # also upgrades a group that was created by importing an .onnx/.engine
        group.update(model_info(pt), from_pt=True, imported=False)
        write_json(gdir / "model.json", group)
    write_json(gdir / "pt_fp32.json", variant_meta("pt_fp32", pt, a.imgsz, built_on=device_model()))
    print(f"[model] {group.get('arch')} task={group.get('task')} classes={group.get('num_classes')}", flush=True)

    trt_version = None
    try:
        import tensorrt
        trt_version = tensorrt.__version__
    except Exception:
        pass

    from ultralytics import YOLO
    for i, variant in enumerate(a.variants, 1):
        prog.update(step=i, current=variant)
        write_json(prog_path, prog)
        ext, kw = VARIANTS[variant]
        dst = gdir / f"{variant}{ext}"
        try:
            if kw is None:
                prog["done"].append(variant)
                continue
            if kw["format"] == "engine" and (trt_version is None or not cuda_ok()):
                raise RuntimeError("TensorRT engines need CUDA + TensorRT (run this on the Jetson)")
            # Each export works on its own copy: Ultralytics writes model.onnx next to the source,
            # and engine export would otherwise overwrite the ONNX variant.
            work = gdir / "_work" / variant
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            shutil.copy(pt, work / "model.pt")
            print(f"[convert] {variant} imgsz={a.imgsz} ...", flush=True)
            t = time.time()
            produced = YOLO(str(work / "model.pt")).export(**kw, imgsz=a.imgsz, batch=1,
                                                           device=0 if cuda_ok() else "cpu")
            shutil.move(str(produced), dst)
            build_s = round(time.time() - t, 1)
            write_json(gdir / f"{variant}.json", variant_meta(
                variant, dst, a.imgsz, build_s=build_s, built_on=device_model(),
                tensorrt=trt_version if kw["format"] == "engine" else None))
            shutil.rmtree(work, ignore_errors=True)
            prog["done"].append(variant)
            print(f"[ok] {variant} -> {dst.name} in {build_s}s", flush=True)
        except Exception as e:  # keep going: one failed format should not stop the others
            prog["failed"][variant] = str(e)[:300]
            print(f"[FAIL] {variant}: {e}", flush=True)
    shutil.rmtree(gdir / "_work", ignore_errors=True)
    prog.update(current="finished")
    write_json(prog_path, prog)
    sys.exit(1 if prog["failed"] else 0)


# ----------------------------------------------------------------------------- system monitor
class SysMonitor(threading.Thread):
    """Samples power / GPU load / temperature with tegrastats (Jetson) or CPU / RAM with psutil."""

    def __init__(self, interval_ms=500):
        super().__init__(daemon=True)
        self.interval = interval_ms
        self.samples = []
        self.t0 = time.perf_counter()
        self.mark_t = 0.0
        self.proc = None
        self.source = "tegrastats" if shutil.which("tegrastats") else "psutil"
        self._halt = threading.Event()

    def run(self):
        if self.source == "tegrastats":
            try:
                self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval)],
                                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                for line in self.proc.stdout:
                    if self._halt.is_set():
                        break
                    s = self.parse(line)
                    if s:
                        self.samples.append(s)
                return
            except Exception:
                self.source = "psutil"
        import psutil
        psutil.cpu_percent()
        while not self._halt.wait(self.interval / 1000):
            self.samples.append({"t": self.t(), "cpu_pct": psutil.cpu_percent(),
                                 "ram_mb": round(psutil.virtual_memory().used / 1e6)})

    def t(self):
        return round(time.perf_counter() - self.t0, 2)

    def parse(self, line):
        s = {"t": self.t()}
        if m := re.search(r"RAM (\d+)/\d+MB", line):
            s["ram_mb"] = int(m.group(1))
        if m := re.search(r"GR3D_FREQ (\d+)%", line):
            s["gpu_pct"] = int(m.group(1))
        if m := re.search(r"\bgpu@([\d.]+)C", line, re.I):
            s["gpu_temp_c"] = float(m.group(1))
        if m := re.search(r"\btj@([\d.]+)C", line, re.I):
            s["tj_temp_c"] = float(m.group(1))
        if m := re.search(r"(?:VDD_IN|POM_5V_IN) (\d+)mW", line):
            s["power_w"] = round(int(m.group(1)) / 1000, 2)
        if m := re.search(r"CPU \[([^\]]+)\]", line):
            loads = [int(x) for x in re.findall(r"(\d+)%@", m.group(1))]
            if loads:
                s["cpu_pct"] = round(sum(loads) / len(loads), 1)
        return s if len(s) > 1 else None

    def mark(self):
        """Call when warm-up ends; only later samples count in the summary."""
        self.mark_t = self.t()

    def stop(self):
        self._halt.set()
        if self.proc:
            self.proc.terminate()

    def summary(self):
        ss = [s for s in self.samples if s["t"] >= self.mark_t] or self.samples
        out = {"source": self.source, "samples": len(ss)}

        def agg(key, fn, name):
            vals = [s[key] for s in ss if key in s]
            if vals:
                out[name] = round(fn(vals), 2)

        agg("power_w", st.mean, "power_w_mean")
        agg("power_w", max, "power_w_max")
        agg("gpu_pct", st.mean, "gpu_util_mean")
        agg("gpu_temp_c", max, "gpu_temp_max")
        agg("tj_temp_c", max, "tj_temp_max")
        agg("cpu_pct", st.mean, "cpu_util_mean")
        agg("ram_mb", max, "ram_mb_max")
        return out

    def save_csv(self, path):
        keys = ["t", "power_w", "gpu_pct", "gpu_temp_c", "tj_temp_c", "cpu_pct", "ram_mb"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.samples)


# --------------------------------------------------------------------------------- video output
_H264 = None


def h264_available():
    global _H264
    if _H264 is None:
        ff = shutil.which("ffmpeg")
        _H264 = bool(ff) and "libx264" in run_out([ff, "-hide_banner", "-encoders"])
    return _H264


class VideoSink:
    """H.264 via ffmpeg (plays in the browser) or OpenCV mp4v as a fallback (download only)."""

    def __init__(self, path, fps, w, h):
        self.proc = self.writer = None
        if h264_available():
            self.codec = "h264"
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", f"{fps:.3f}", "-i", "-", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                 "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE)
        else:
            import cv2
            self.codec = "mp4v"
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, img):
        if self.proc:
            self.proc.stdin.write(img.tobytes())
        else:
            self.writer.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.writer:
            self.writer.release()


def draw_banner(img, text):
    import cv2
    scale = max(0.5, img.shape[1] / 1600)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    pad = int(8 * scale)
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, th + 2 * pad), (20, 20, 20), -1)
    cv2.putText(img, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------------------- infer
def pct(vals, q):
    import numpy as np
    return round(float(np.percentile(vals, q)), 3) if vals else None


def stats(vals):
    if not vals:
        return {}
    return {"mean": round(st.mean(vals), 3), "std": round(st.pstdev(vals), 3), "p50": pct(vals, 50),
            "p95": pct(vals, 95), "min": round(min(vals), 3), "max": round(max(vals), 3)}


def frame_boxes(r):
    if r.boxes is None or len(r.boxes) == 0:
        return []
    cls = r.boxes.cls.tolist()
    conf = r.boxes.conf.tolist()
    xyxy = r.boxes.xyxy.tolist()
    return [[int(c), round(p, 3), *[round(v, 1) for v in b]] for c, p, b in zip(cls, conf, xyxy)]


def onnx_providers(model):
    try:
        return model.predictor.model.session.get_providers()
    except Exception:
        return None


def cmd_infer(a):
    import cv2
    from ultralytics import YOLO

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(read_text(Path(a.model).with_suffix(".json")) or "{}")
    device = pick_device(a.device, a.model)
    model = YOLO(a.model, task=a.task or None)
    names = model.names
    label = a.label or Path(a.model).name
    kw = dict(imgsz=a.imgsz, conf=a.conf, iou=a.iou, device=device, verbose=False)
    is_image = Path(a.source).suffix.lower() in IMG_EXT
    t_start = time.time()

    summary = {"label": label, "model": {"path": a.model, **meta}, "created": now(),
               "settings": {"imgsz": a.imgsz, "conf": a.conf, "iou": a.iou, "device": str(device),
                            "warmup": a.warmup, "max_frames": a.max_frames, "save": a.save}}
    mon = SysMonitor()
    mon.start()

    if is_image:
        img = cv2.imread(a.source)
        if img is None:
            raise SystemExit(f"cannot read image {a.source}")
        for _ in range(a.warmup):
            model.predict(img, **kw)
        mon.mark()
        speeds = []
        for _ in range(max(1, a.repeats)):
            r = model.predict(img, **kw)[0]
            speeds.append(r.speed)
        boxes = frame_boxes(r)
        if a.save:
            cv2.imwrite(str(out / "output.jpg"), r.plot())
        summary["source"] = {"path": a.source, "type": "image", "width": img.shape[1], "height": img.shape[0]}
        frames_measured = speeds
        pipeline_fps = None
        with open(out / "detections.jsonl", "w") as f:
            f.write(json.dumps({"f": 0, "b": boxes}) + "\n")
        summary["boxes"] = [{"class": names[b[0]], "conf": b[1], "xyxy": b[2:]} for b in boxes[:300]]
        per_frame_dets = [len(boxes)]
        all_boxes = boxes
        summary["outputs"] = {"image": "output.jpg" if a.save else None}
    else:
        source = int(a.source) if a.source.isdigit() else a.source
        cap = cv2.VideoCapture(source)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        if not 1 <= fps_in <= 240:
            fps_in = 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if a.max_frames:
            total = min(total, a.max_frames) if total > 0 else a.max_frames
        summary["source"] = {"path": str(a.source), "type": "video", "fps": round(fps_in, 2),
                             "width": w, "height": h, "frames_in_file": total}

        class_keys = [names[k] for k in sorted(names)]
        fcsv = open(out / "frames.csv", "w", newline="")
        fieldnames = ["frame", "time_s", "warmup", "preprocess_ms", "inference_ms", "postprocess_ms",
                      "latency_ms", "loop_ms", "detections", "mean_conf"] + [f"n_{c}" for c in class_keys]
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        fdet = open(out / "detections.jsonl", "w")
        sink = None
        rows, per_frame_dets, all_boxes = [], [], []
        frames_measured, n, t_measure, t_prev = [], 0, None, time.perf_counter()
        recent = []
        for r in model.predict(source, stream=True, **kw):
            t_now = time.perf_counter()
            loop_ms = (t_now - t_prev) * 1000
            t_prev = t_now
            warm = n < a.warmup
            if not warm and t_measure is None:
                t_measure = t_now - loop_ms / 1000  # measured period starts with this frame
                mon.mark()
            sp = r.speed
            lat = sp.get("preprocess", 0) + sp.get("inference", 0) + sp.get("postprocess", 0)
            boxes = frame_boxes(r)
            counts = {}
            for b in boxes:
                counts[names[b[0]]] = counts.get(names[b[0]], 0) + 1
            row = {"frame": n, "time_s": round(n / fps_in, 3), "warmup": int(warm),
                   "preprocess_ms": round(sp.get("preprocess", 0), 3), "inference_ms": round(sp.get("inference", 0), 3),
                   "postprocess_ms": round(sp.get("postprocess", 0), 3), "latency_ms": round(lat, 3),
                   "loop_ms": round(loop_ms, 3), "detections": len(boxes),
                   "mean_conf": round(st.mean(b[1] for b in boxes), 3) if boxes else "",
                   **{f"n_{c}": counts.get(c, 0) for c in class_keys}}
            writer.writerow(row)
            fdet.write(json.dumps({"f": n, "b": boxes}) + "\n")
            if not warm:
                frames_measured.append(sp)
                per_frame_dets.append(len(boxes))
                all_boxes.extend(boxes)
            if a.save:
                img = r.plot()
                recent = (recent + [lat])[-30:]
                draw_banner(img, f"{label} | {st.mean(recent):.1f} ms | {1000 / max(st.mean(recent), 1e-6):.0f} FPS")
                if sink is None:
                    sink = VideoSink(out / "output.mp4", fps_in, img.shape[1], img.shape[0])
                sink.write(img)
            n += 1
            if n % 10 == 0:
                write_json(out / "progress.json", {"frame": n, "total": total})
            if a.max_frames and n >= a.max_frames:
                break
        wall = time.perf_counter() - t_measure if t_measure else 0
        fcsv.close()
        fdet.close()
        if sink:
            sink.close()
        if not n:
            raise SystemExit(f"no frames could be read from {a.source}")
        pipeline_fps = round(len(frames_measured) / wall, 2) if wall > 0 else None
        summary["frames"] = n
        summary["outputs"] = {"video": "output.mp4" if sink else None, "video_codec": sink.codec if sink else None,
                              "frames_csv": "frames.csv"}

    mon.stop()
    mon.save_csv(out / "system.csv")
    pre = [s.get("preprocess", 0) for s in frames_measured]
    inf = [s.get("inference", 0) for s in frames_measured]
    post = [s.get("postprocess", 0) for s in frames_measured]
    lat = [p + i + q for p, i, q in zip(pre, inf, post)]
    per_class = {}
    for b in all_boxes:
        per_class[names[b[0]]] = per_class.get(names[b[0]], 0) + 1
    summary.update({
        "measured_frames": len(frames_measured),
        "preprocess_ms": stats(pre), "inference_ms": stats(inf), "postprocess_ms": stats(post), "latency_ms": stats(lat),
        "fps_model": round(1000 / st.mean(lat), 2) if lat and st.mean(lat) > 0 else None,
        "fps_pipeline": pipeline_fps,
        "detections": {"total": len(all_boxes), "per_frame_mean": round(st.mean(per_frame_dets), 3) if per_frame_dets else 0,
                       "per_class": per_class,
                       "mean_conf": round(st.mean(b[1] for b in all_boxes), 4) if all_boxes else None},
        "system": mon.summary(),
        "runtime": {"device": str(device), "onnx_providers": onnx_providers(model) if a.model.endswith(".onnx") else None},
        "elapsed_s": round(time.time() - t_start, 1),
    })
    if summary["system"].get("power_w_mean") and summary["fps_model"]:
        summary["fps_per_watt"] = round(summary["fps_model"] / summary["system"]["power_w_mean"], 2)
    write_json(out / "summary.json", summary)
    write_json(out / "progress.json", {"frame": summary.get("frames", 1), "total": summary.get("frames", 1), "done": True})
    print(f"[done] {label}: latency p50 {summary['latency_ms'].get('p50')} ms, model FPS {summary['fps_model']}, "
          f"pipeline FPS {pipeline_fps}, detections {len(all_boxes)}", flush=True)


# ----------------------------------------------------------------------------------------- val
def cmd_val(a):
    from ultralytics import YOLO
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device, a.model)
    m = YOLO(a.model, task=a.task or None)
    print(f"[val] {a.model} on {a.data} ({a.split})", flush=True)
    v = m.val(data=a.data, split=a.split, imgsz=a.imgsz, batch=1, device=device, plots=False, verbose=False,
              project=str(out), name="val", exist_ok=True)
    names = v.names
    res = {"data": a.data, "split": a.split, "map50": round(float(v.box.map50), 4), "map50_95": round(float(v.box.map), 4),
           "precision": round(float(v.box.mp), 4), "recall": round(float(v.box.mr), 4),
           "per_class_ap50": {names[int(c)]: round(float(ap), 4) for c, ap in zip(v.box.ap_class_index, v.box.ap50)},
           "speed_ms": {k: round(x, 3) for k, x in v.speed.items()}}
    write_json(out / "val.json", res)
    print(f"[val] mAP50 {res['map50']}  mAP50-95 {res['map50_95']}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("env")
    e.add_argument("--out", required=True)

    c = sub.add_parser("convert")
    c.add_argument("--group-dir", required=True, help="folder holding pt_fp32.pt; outputs are written next to it")
    c.add_argument("--variants", nargs="*", default=[], choices=list(VARIANTS))
    c.add_argument("--imgsz", type=int, default=640)

    for name in ("infer", "val"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--task", default="")
        s.add_argument("--out", required=True)
        s.add_argument("--imgsz", type=int, default=640)
        s.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    i = sub.choices["infer"]
    i.add_argument("--source", required=True, help="video / image file, camera index or rtsp:// url")
    i.add_argument("--label", default="")
    i.add_argument("--conf", type=float, default=0.25)
    i.add_argument("--iou", type=float, default=0.7)
    i.add_argument("--warmup", type=int, default=10, help="frames excluded from the statistics")
    i.add_argument("--repeats", type=int, default=30, help="timed runs for a single image")
    i.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    i.add_argument("--save", action="store_true", help="write the annotated video / image")
    v = sub.choices["val"]
    v.add_argument("--data", required=True)
    v.add_argument("--split", default="val")

    a = p.parse_args()
    {"env": cmd_env, "convert": cmd_convert, "infer": cmd_infer, "val": cmd_val}[a.cmd](a)


if __name__ == "__main__":
    main()

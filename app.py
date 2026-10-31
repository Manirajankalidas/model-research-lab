#!/usr/bin/env python3
"""YOLO Model Lab: convert, run and compare YOLO models on a Jetson from the browser.

  pip install flask
  python app.py --host 0.0.0.0 --port 8000      # then open http://<jetson-ip>:8000

Heavy work (torch / TensorRT) runs in worker.py subprocesses, one job at a time.
Everything is stored under data/: models/<name>/, videos/, jobs/<id>/.
"""
import argparse
import ast
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

import analysis

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS, VIDEOS, JOBS = DATA / "models", DATA / "videos", DATA / "jobs"
ENV_FILE = DATA / "env.json"
WORKER = HERE / "worker.py"

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VARIANT_LABEL = {"pt_fp32": "PyTorch FP32", "onnx_fp32": "ONNX FP32", "trt_fp32": "TensorRT FP32",
                 "trt_fp16": "TensorRT FP16", "trt_int8": "TensorRT INT8"}
VARIANT_EXT = {"pt_fp32": ".pt", "onnx_fp32": ".onnx", "trt_fp32": ".engine", "trt_fp16": ".engine", "trt_int8": ".engine"}
CONVERTIBLE = ["onnx_fp32", "trt_fp32", "trt_fp16"]  # trt_int8: on hold

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024**3  # 4 GB uploads (videos)


# ------------------------------------------------------------------------------------- helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def err(msg, code=400):
    return jsonify(error=msg), code


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", (s or "").strip()).strip("._")[:60]


def inside(base, path):
    """Resolve a user-supplied relative path and make sure it stays inside base."""
    p = (base / path).resolve()
    return p if base.resolve() in p.parents and p.exists() else None


def num(name, kind, lo, hi, default):
    raw = request.form.get(name, "").strip()
    if raw == "":
        return default
    try:
        v = kind(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number")
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return v


def dir_mb(path):
    return round(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6, 1) if Path(path).exists() else 0


def save_upload(fs, dst):
    """Save an uploaded file and return its sha256."""
    h = hashlib.sha256()
    with open(dst, "wb") as f:
        while chunk := fs.stream.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------- model library
def group_busy(name):
    return any(j["state"] in ("queued", "running") and name in j.get("groups", []) for j in jobs.all())


def list_models():
    groups = []
    for gdir in sorted(p for p in MODELS.glob("*") if p.is_dir()):
        info = read_json(gdir / "model.json", {})
        variants = []
        for v, label in VARIANT_LABEL.items():
            f = gdir / f"{v}{VARIANT_EXT[v]}"
            if not f.exists():
                continue
            meta = read_json(gdir / f"{v}.json", {})
            variants.append({**meta, "key": f"{gdir.name}/{v}", "variant": v, "label": label, "file": f.name,
                             "size_mb": meta.get("size_mb") or round(f.stat().st_size / 1e6, 2),
                             "imgsz": meta.get("imgsz") or info.get("train_imgsz") or 640})
        groups.append({"name": gdir.name, "info": info, "variants": variants, "busy": group_busy(gdir.name)})
    return groups


def resolve_model(key, imgsz_override=None):
    """'group/variant' -> dict(key, label, path, task, imgsz, group). Raises ValueError."""
    group, _, variant = (key or "").partition("/")
    if variant not in VARIANT_LABEL or safe_name(group) != group:
        raise ValueError(f"Unknown model: {key}")
    path = MODELS / group / f"{variant}{VARIANT_EXT[variant]}"
    if not path.exists():
        raise ValueError(f"Model file is missing: {key}")
    info = read_json(MODELS / group / "model.json", {})
    meta = read_json(MODELS / group / f"{variant}.json", {})
    imgsz = meta.get("imgsz") or info.get("train_imgsz") or 640
    if variant == "pt_fp32" and imgsz_override:  # exported models have a fixed input size
        imgsz = imgsz_override
    return {"key": key, "group": group, "variant": variant, "label": f"{group} · {VARIANT_LABEL[variant]}",
            "path": str(path), "task": info.get("task", ""), "imgsz": int(imgsz)}


def exported_meta(path):
    """Ultralytics stores names / imgsz / task inside exported ONNX and engine files."""
    path = Path(path)
    try:
        if path.suffix == ".engine":
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(4), byteorder="little")
                if 0 < n < 1_000_000:
                    return json.loads(f.read(n).decode("utf-8"))
        elif path.suffix == ".onnx":
            import onnx
            m = onnx.load(str(path), load_external_data=False)
            meta = {p.key: p.value for p in m.metadata_props}
            for k in ("imgsz", "names", "args"):
                if k in meta:
                    meta[k] = ast.literal_eval(meta[k])
            return meta
    except Exception:
        pass
    return {}


# -------------------------------------------------------------------------------- video library
def video_info(path):
    side = path.with_name(path.name + ".json")
    info = read_json(side)
    if info is None:
        info = {}
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info = {"fps": round(fps, 2), "frames": frames, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "duration_s": round(frames / fps, 1) if fps else None}
            cap.release()
        except Exception:
            pass
        write_json(side, info)
    return {"name": path.name, "size_mb": round(path.stat().st_size / 1e6, 1), **info}


def list_videos():
    return [video_info(p) for p in sorted(VIDEOS.glob("*"), key=lambda p: -p.stat().st_mtime)
            if p.suffix.lower() in VIDEO_EXT]


def store_video(fs):
    stem, ext = safe_name(Path(fs.filename).stem) or "video", Path(fs.filename).suffix.lower()
    dst, i = VIDEOS / f"{stem}{ext}", 1
    while dst.exists():
        dst, i = VIDEOS / f"{stem}_{i}{ext}", i + 1
    save_upload(fs, dst)
    return dst


def resolve_source(jdir):
    """Reads source_type + video_name / source_file / source_path from the form. Returns (source, display, is_stream)."""
    kind = request.form.get("source_type", "library")
    if kind == "library":
        name = request.form.get("video_name", "")
        p = inside(VIDEOS, name) if name else None
        if not p:
            raise ValueError("Choose a video from the library")
        return str(p), p.name, False
    if kind == "upload":
        fs = request.files.get("source_file")
        if not fs or not fs.filename:
            raise ValueError("Choose a video or image file to upload")
        ext = Path(fs.filename).suffix.lower()
        if ext in VIDEO_EXT:
            p = store_video(fs)  # kept in the library so it can be reused for comparisons
            return str(p), p.name, False
        if ext in IMG_EXT:
            p = jdir / f"input{ext}"
            save_upload(fs, p)
            return str(p), Path(fs.filename).name, False
        raise ValueError("File must be a video (mp4, avi, mov, mkv, webm) or an image (jpg, png, bmp)")
    if kind == "path":
        src = request.form.get("source_path", "").strip()
        if src.isdigit() or re.match(r"^(rtsp|rtmp|http|https)://", src):
            return src, src, True
        if not Path(src).is_file():
            raise ValueError(f"File not found on this device: {src}")
        return src, Path(src).name, False
    raise ValueError("Unknown source type")


# ------------------------------------------------------------------------------------------ jobs
class Cancelled(Exception):
    pass


class JobManager:
    """Persistent FIFO queue. One job runs at a time because the Jetson has one GPU and limited memory."""

    def __init__(self):
        self.jobs, self.q, self.lock = {}, queue.Queue(), threading.Lock()
        self.proc = None
        for f in sorted(JOBS.glob("*/job.json"), key=lambda p: p.stat().st_mtime):
            j = read_json(f)
            if not j:
                continue
            if j["state"] == "running":
                j.update(state="failed", error="Interrupted: the server was restarted", finished=now())
                self.save(j)
            self.jobs[j["id"]] = j
            if j["state"] == "queued":
                self.q.put(j["id"])
        threading.Thread(target=self.loop, daemon=True).start()

    def all(self):
        with self.lock:
            return list(self.jobs.values())

    def get(self, jid):
        return self.jobs.get(jid)

    def save(self, j):
        write_json(JOBS / j["id"] / "job.json", j)

    def create(self, kind, title, params, groups, jid=None):
        jid = jid or new_job_dir()
        j = {"id": jid, "kind": kind, "title": title, "state": "queued", "created": now(), "started": None,
             "finished": None, "params": params, "groups": groups, "steps": [], "error": None, "warnings": []}
        with self.lock:
            self.jobs[jid] = j
        self.save(j)
        self.q.put(jid)
        return j

    def cancel(self, j):
        j["cancel"] = True
        if j["state"] == "queued":
            j.update(state="cancelled", finished=now())
            self.save(j)
        elif j["state"] == "running" and self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)  # also stops tegrastats / ffmpeg children
            except ProcessLookupError:
                pass

    def delete(self, j):
        with self.lock:
            self.jobs.pop(j["id"], None)
        shutil.rmtree(JOBS / j["id"], ignore_errors=True)

    def loop(self):
        while True:
            jid = self.q.get()
            j = self.jobs.get(jid)
            if not j or j["state"] != "queued":
                continue
            j.update(state="running", started=now())
            self.save(j)
            try:
                getattr(self, "run_" + j["kind"])(j)
                j["state"] = "failed" if j.get("error") else "done"
            except Cancelled:
                j.update(state="cancelled", error="Cancelled by user")
            except Exception as e:  # never let one job kill the queue
                j.update(state="failed", error=f"{type(e).__name__}: {e}")
            j["finished"] = now()
            self.proc = None
            self.save(j)

    def call(self, j, args, step):
        if j.get("cancel"):
            raise Cancelled()
        step["state"] = "running"
        self.save(j)
        cmd = [sys.executable, str(WORKER), *map(str, args)]
        with open(JOBS / j["id"] / "log.txt", "a") as log:
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=JOBS / j["id"],
                                         start_new_session=True)
            rc = self.proc.wait()
        if j.get("cancel"):
            step["state"] = "cancelled"
            raise Cancelled()
        step["state"] = "done" if rc == 0 else "failed"
        self.save(j)
        return rc

    def run_convert(self, j):
        p = j["params"]
        gdir = MODELS / p["group"]
        step = {"state": "pending", "label": "Convert " + (", ".join(VARIANT_LABEL[v] for v in p["variants"]) or "(read model info)")}
        j["steps"] = [step]
        self.call(j, ["convert", "--group-dir", gdir, "--imgsz", p["imgsz"], "--variants", *p["variants"]], step)
        prog = read_json(gdir / "progress.json", {})
        failed = prog.get("failed", {})
        j["result"] = {"done": prog.get("done", []), "failed": failed}
        if failed:
            j["warnings"] = [f"{VARIANT_LABEL[k]}: {v}" for k, v in failed.items()]
            if len(failed) == len(p["variants"]):
                j["error"] = "No format could be built"

    def run_bench(self, j):
        """Inference (one model) and comparison (several models) share this runner."""
        p = j["params"]
        jdir = JOBS / j["id"]
        steps = [{"label": f"Run {m['label']}", "state": "pending", "model": i, "dir": f"results/{i}"} for i, m in enumerate(p["models"])]
        if p.get("data_yaml"):
            steps += [{"label": f"mAP {m['label']}", "state": "pending", "model": i, "val": True} for i, m in enumerate(p["models"])]
        j["steps"] = steps
        s = p["settings"]
        for step in steps:
            m = p["models"][step["model"]]
            out = jdir / "results" / str(step["model"])
            common = ["--model", m["path"], "--task", m["task"], "--out", out, "--imgsz", m["imgsz"], "--device", s["device"]]
            if step.get("val"):
                rc = self.call(j, ["val", *common, "--data", p["data_yaml"], "--split", p["split"]], step)
            else:
                args = ["infer", *common, "--source", p["source"], "--label", m["label"], "--conf", s["conf"],
                        "--iou", s["iou"], "--warmup", s["warmup"], "--max-frames", s["max_frames"]]
                rc = self.call(j, args + (["--save"] if s["save"] else []), step)
            if rc != 0:
                j["warnings"].append(f"{step['label']} failed (see log)")
            analysis.build(jdir, p["models"], p.get("baseline", 0), read_json(ENV_FILE))
        if all(st.get("state") == "failed" for st in steps if not st.get("val")):
            j["error"] = "All model runs failed (see log)"

    run_infer = run_compare = run_bench


def new_job_dir():
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    (JOBS / jid).mkdir(parents=True)
    return jid


# ------------------------------------------------------------------------------------- env info
def refresh_env():
    subprocess.run([sys.executable, str(WORKER), "env", "--out", str(ENV_FILE)], capture_output=True, timeout=300)


# ----------------------------------------------------------------------------------------- pages
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/env")
def api_env():
    env = read_json(ENV_FILE)
    du = shutil.disk_usage(DATA)
    running = next((j for j in jobs.all() if j["state"] == "running"), None)
    return jsonify(env=env, disk_free_gb=round(du.free / 1e9, 1),
                   queue={"running": running and {"id": running["id"], "title": running["title"], "kind": running["kind"]},
                          "queued": sum(j["state"] == "queued" for j in jobs.all())},
                   storage={"models_mb": dir_mb(MODELS), "videos_mb": dir_mb(VIDEOS), "jobs_mb": dir_mb(JOBS)})


@app.post("/api/env/refresh")
def api_env_refresh():
    threading.Thread(target=refresh_env, daemon=True).start()
    return jsonify(ok=True)


# ---------------------------------------------------------------------------------------- models
@app.get("/api/models")
def api_models():
    return jsonify(list_models())


@app.post("/api/convert")
def api_convert():
    variants = [v for v in request.form.getlist("variants") if v in CONVERTIBLE]
    rebuild = request.form.get("rebuild") == "on"
    try:
        imgsz = num("imgsz", int, 32, 4096, 640)
        if imgsz % 32:
            raise ValueError("Image size must be a multiple of 32")
    except ValueError as e:
        return err(str(e))

    if request.form.get("source") == "existing":
        name = request.form.get("group", "")
        gdir = MODELS / name
        if not name or safe_name(name) != name or not (gdir / "pt_fp32.pt").exists():
            return err("Choose a model from the library that has a .pt file")
    else:
        fs = request.files.get("weights")
        if not fs or not fs.filename.lower().endswith(".pt"):
            return err("Upload a .pt file")
        name = safe_name(request.form.get("name") or Path(fs.filename).stem)
        if not name:
            return err("Give the model a name")
        gdir = MODELS / name
        if group_busy(name):
            return err(f"'{name}' is being processed by another job; wait for it to finish")
        MODELS.mkdir(parents=True, exist_ok=True)
        tmp = MODELS / f".upload-{uuid.uuid4().hex}.pt"
        digest = save_upload(fs, tmp)
        if (gdir / "pt_fp32.pt").exists():
            if sha256(gdir / "pt_fp32.pt") != digest:
                tmp.unlink()
                return err(f"A different model is already called '{name}'. Choose another name, or delete it first.")
            tmp.unlink()  # same weights uploaded again: just add formats
        else:
            gdir.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, gdir / "pt_fp32.pt")

    if group_busy(name):
        return err(f"'{name}' is being processed by another job; wait for it to finish")
    existing = {v["variant"]: v for v in next((g["variants"] for g in list_models() if g["name"] == name), [])}
    todo = [v for v in variants if rebuild or v not in existing or existing[v].get("imgsz") != imgsz]
    if variants and not todo:
        return err("All selected formats already exist at this image size. Tick 'Rebuild' to build them again.")
    title = f"Convert {name} → " + (", ".join(VARIANT_LABEL[v] for v in todo) or "library only")
    j = jobs.create("convert", title, {"group": name, "variants": todo, "imgsz": imgsz}, [name])
    return jsonify(id=j["id"])


@app.post("/api/models/import")
def api_import():
    """Add an existing .onnx / .engine file to the library without converting."""
    fs = request.files.get("file")
    if not fs or Path(fs.filename).suffix.lower() not in (".onnx", ".engine"):
        return err("Choose an .onnx or .engine file (use Convert for .pt files)")
    name = safe_name(request.form.get("name") or Path(fs.filename).stem)
    if not name:
        return err("Give the model a name")
    if group_busy(name):
        return err(f"'{name}' is being processed by another job")
    gdir = MODELS / name
    gdir.mkdir(parents=True, exist_ok=True)
    ext = Path(fs.filename).suffix.lower()
    tmp = gdir / f".import{ext}"
    save_upload(fs, tmp)
    meta = exported_meta(tmp)
    args = meta.get("args") or {}
    if ext == ".onnx":
        variant = "onnx_fp32"
    else:
        prec = request.form.get("precision", "auto")
        if prec == "auto":
            prec = "int8" if args.get("int8") else "fp16" if args.get("half") else "fp32" if meta else ""
        if prec not in ("fp32", "fp16", "int8"):
            tmp.unlink()
            return err("Could not detect the engine precision; choose it in the form")
        variant = f"trt_{prec}"
    dst = gdir / f"{variant}{ext}"
    if dst.exists() and request.form.get("rebuild") != "on":
        tmp.unlink()
        return err(f"'{name}' already has a {VARIANT_LABEL[variant]} file. Tick 'Replace' to overwrite it.")
    os.replace(tmp, dst)
    imgsz = meta.get("imgsz")
    imgsz = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
    try:
        imgsz = int(request.form.get("imgsz") or imgsz or 640)
    except ValueError:
        imgsz = 640
    info = read_json(gdir / "model.json", {})
    if "task" not in info:
        names = meta.get("names") or {}
        info.update(task=meta.get("task", "detect"), names=names, num_classes=len(names), arch=None, imported=True)
        write_json(gdir / "model.json", info)
    write_json(gdir / f"{variant}.json", {
        "variant": variant, "format": VARIANT_LABEL[variant].split()[0], "precision": variant.split("_")[1].upper(),
        "file": dst.name, "imgsz": imgsz, "size_mb": round(dst.stat().st_size / 1e6, 2), "created": now(),
        "built_on": "imported", "imported": True})
    return jsonify(ok=True, key=f"{name}/{variant}")


@app.delete("/api/models/<group>")
@app.delete("/api/models/<group>/<variant>")
def api_model_delete(group, variant=None):
    gdir = MODELS / group
    if safe_name(group) != group or not gdir.is_dir():
        return err("Unknown model", 404)
    if group_busy(group):
        return err("A job is using this model; wait for it to finish")
    if variant is None:
        shutil.rmtree(gdir)
    elif variant in VARIANT_LABEL:
        for f in (gdir / f"{variant}{VARIANT_EXT[variant]}", gdir / f"{variant}.json"):
            f.unlink(missing_ok=True)
        if not any(gdir.glob("*.pt")) and not any(gdir.glob("*.onnx")) and not any(gdir.glob("*.engine")):
            shutil.rmtree(gdir)
    return jsonify(ok=True)


@app.get("/files/models/<group>/<name>")
def file_model(group, name):
    return send_from_directory(MODELS / safe_name(group), name, as_attachment=True,
                               download_name=f"{safe_name(group)}_{name}")


# ---------------------------------------------------------------------------------------- videos
@app.get("/api/videos")
def api_videos():
    return jsonify(list_videos())


@app.post("/api/videos")
def api_video_upload():
    fs = request.files.get("file")
    if not fs or Path(fs.filename).suffix.lower() not in VIDEO_EXT:
        return err("Choose a video file (mp4, avi, mov, mkv, webm)")
    VIDEOS.mkdir(parents=True, exist_ok=True)
    return jsonify(video_info(store_video(fs)))


@app.delete("/api/videos/<name>")
def api_video_delete(name):
    p = inside(VIDEOS, name)
    if not p:
        return err("Unknown video", 404)
    p.unlink()
    p.with_name(p.name + ".json").unlink(missing_ok=True)
    return jsonify(ok=True)


@app.get("/files/videos/<name>")
def file_video(name):
    return send_from_directory(VIDEOS, name)


# ------------------------------------------------------------------------------ inference / compare
def bench_settings():
    return {"conf": num("conf", float, 0.01, 0.99, 0.25), "iou": num("iou", float, 0.1, 0.95, 0.7),
            "warmup": num("warmup", int, 0, 500, 10), "max_frames": num("max_frames", int, 0, 10**7, 0),
            "device": request.form.get("device") if request.form.get("device") in ("auto", "gpu", "cpu") else "auto",
            "save": request.form.get("save") == "on"}


def start_bench(kind, keys):
    jid = new_job_dir()
    jdir = JOBS / jid
    try:
        s = bench_settings()
        imgsz = num("imgsz", int, 32, 4096, 0) or None
        models = [resolve_model(k, imgsz) for k in keys]
        if not models:
            raise ValueError("Choose at least one model")
        busy = [m["group"] for m in models if any(j["state"] in ("queued", "running") and j["kind"] == "convert"
                                                 and m["group"] in j["groups"] for j in jobs.all())]
        if busy:
            raise ValueError(f"'{busy[0]}' is still being converted; wait for it to finish")
        source, display, stream = resolve_source(jdir)
        if stream and not s["max_frames"]:
            raise ValueError("Camera / stream sources need 'Max frames' (otherwise the run never ends)")
        data_yaml = request.form.get("data_yaml", "").strip()
        if data_yaml and not Path(data_yaml).is_file():
            raise ValueError(f"Dataset YAML not found on this device: {data_yaml}")
    except ValueError as e:
        shutil.rmtree(jdir, ignore_errors=True)
        return err(str(e))
    baseline = request.form.get("baseline", "")
    base_i = next((i for i, m in enumerate(models) if m["key"] == baseline), 0)
    title = request.form.get("title", "").strip() or (
        f"{models[0]['label']} on {display}" if kind == "infer" else f"{len(models)} models on {display}")
    params = {"models": models, "source": source, "source_name": display, "settings": s, "baseline": base_i,
              "data_yaml": data_yaml or None, "split": request.form.get("split", "val"), "notes": request.form.get("notes", "")}
    j = jobs.create(kind, title[:120], params, sorted({m["group"] for m in models}), jid=jid)
    return jsonify(id=j["id"])


@app.post("/api/infer")
def api_infer():
    return start_bench("infer", [request.form.get("model", "")])


@app.post("/api/compare")
def api_compare():
    return start_bench("compare", request.form.getlist("models"))


@app.get("/api/jobs")
def api_jobs():
    kinds = request.args.get("kind", "").split(",") if request.args.get("kind") else None
    out = [{k: j.get(k) for k in ("id", "kind", "title", "state", "created", "finished", "error")}
           | {"models": len(j["params"].get("models", [])), "source": j["params"].get("source_name")}
           for j in jobs.all() if not kinds or j["kind"] in kinds]
    return jsonify(sorted(out, key=lambda x: x["created"], reverse=True))


@app.get("/api/jobs/<jid>")
def api_job(jid):
    j = jobs.get(jid) or abort(404)
    jdir = JOBS / jid
    log = (jdir / "log.txt").read_text(errors="replace")[-8000:] if (jdir / "log.txt").exists() else ""
    steps = []
    for s in j["steps"]:
        s = dict(s)
        if s.get("state") == "running" and s.get("dir"):
            s["progress"] = read_json(jdir / s["dir"] / "progress.json")
        if s.get("state") == "running" and j["kind"] == "convert":
            s["progress"] = read_json(MODELS / j["params"]["group"] / "progress.json")
        steps.append(s)
    return jsonify({**{k: v for k, v in j.items() if k != "cancel"}, "steps": steps, "log": log,
                    "has_results": (jdir / "comparison.json").exists()})


@app.get("/api/jobs/<jid>/results")
def api_job_results(jid):
    j = jobs.get(jid) or abort(404)
    res = read_json(JOBS / jid / "comparison.json") or abort(404)
    return jsonify({**res, "job": {k: j[k] for k in ("id", "kind", "title", "state", "created", "finished", "params")}})


@app.post("/api/jobs/<jid>/cancel")
def api_job_cancel(jid):
    jobs.cancel(jobs.get(jid) or abort(404))
    return jsonify(ok=True)


@app.delete("/api/jobs/<jid>")
def api_job_delete(jid):
    j = jobs.get(jid) or abort(404)
    if j["state"] in ("queued", "running"):
        return err("Cancel the job before deleting it")
    jobs.delete(j)
    return jsonify(ok=True)


@app.get("/api/jobs/<jid>/export")
def api_job_export(jid):
    """Zip of everything needed for the report: summaries, per-frame CSVs, system logs, optionally videos."""
    j = jobs.get(jid) or abort(404)
    jdir = JOBS / jid
    with_video = request.args.get("videos") == "1"
    zpath = jdir / ("export_videos.zip" if with_video else "export.zip")
    skip = {"export.zip", "export_videos.zip"}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(jdir.rglob("*")):
            rel = f.relative_to(jdir)
            if not f.is_file() or f.name in skip or f.suffix == ".tmp" \
                    or (f.suffix.lower() in VIDEO_EXT | IMG_EXT and not with_video):
                continue
            z.write(f, Path(jid) / rel, compress_type=zipfile.ZIP_STORED if f.suffix.lower() in VIDEO_EXT else None)
    return send_from_directory(jdir, zpath.name, as_attachment=True, download_name=f"{safe_name(j['title'])}_{jid}.zip")


@app.get("/files/jobs/<jid>/<path:sub>")
def file_job(jid, sub):
    if jid not in jobs.jobs:
        abort(404)
    return send_from_directory(JOBS / jid, sub, as_attachment=request.args.get("dl") == "1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 makes it reachable from other PCs (no login!)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    for d in (MODELS, VIDEOS, JOBS):
        d.mkdir(parents=True, exist_ok=True)
    jobs = JobManager()
    threading.Thread(target=refresh_env, daemon=True).start()
    print(f"YOLO Model Lab on http://{a.host}:{a.port}  (data in {DATA})")
    app.run(host=a.host, port=a.port, threaded=True)

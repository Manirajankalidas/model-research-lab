# Model Lab: YOLO edge benchmarking for Jetson

A browser app for converting custom YOLO models (YOLOv8, YOLO11, …) to faster formats, running them, and comparing them side by side on a Jetson Orin Nano. The results are numbers, charts and videos you can put in a research report.

```
your_model.pt ──► ONNX FP32
              ──► TensorRT engine FP32
              ──► TensorRT engine FP16
              (TensorRT INT8: on hold)
```

> **TensorRT engines only work on the GPU and TensorRT version that built them.** Run the app, and every conversion, on the Jetson itself.

---

## Contents

1. [Install on the Jetson](#1-install-on-the-jetson-orin-nano)
2. [Start the app](#2-start-the-app)
3. [The interface](#3-the-interface)
4. [Workflow for the research comparison](#4-workflow-for-the-research-comparison)
5. [What is measured](#5-what-is-measured)
6. [Saved data](#6-saved-data)
7. [How it works and files](#7-how-it-works-and-files)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Install on the Jetson Orin Nano

JetPack already includes CUDA, TensorRT and OpenCV. First install **PyTorch and torchvision from NVIDIA's Jetson wheels** for your JetPack version, then:

```bash
pip install ultralytics --no-deps
pip install onnx onnxslim matplotlib pyyaml tqdm psutil py-cpuinfo pandas seaborn requests scipy flask
sudo apt install ffmpeg          # lets output videos play in the browser
```

**Optional but recommended:** install the **GPU build of ONNX Runtime** for Jetson, from NVIDIA's Jetson Zoo or the Jetson AI Lab pip index. With the normal `pip install onnxruntime`, ONNX models run on the **CPU**. ONNX then looks much slower than it really is, and the app tags those results **ORT CPU**.

Check the setup:

```bash
python3 -c "import torch, tensorrt, cv2; print(torch.cuda.is_available(), tensorrt.__version__, cv2.__version__)"
# expected: True <version> <version>
```

For stable, repeatable numbers, put the Jetson in maximum performance mode before measuring:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

## 2. Start the app

```bash
cd Model_Optimationation
python3 app.py --host 0.0.0.0 --port 8000
```

Open `http://<jetson-ip>:8000` from a PC on the same network. Find the Jetson's IP with `hostname -I`.

- After updating the files in `static/`, reload the browser with **Ctrl+Shift+R**. After updating the `.py` files, restart `app.py`.
- The page uses the Inter font from Google Fonts. If the browsing PC has no internet, it falls back to the system font; everything else works offline.

> **There is no login.** Uploaded `.pt` files are loaded by PyTorch, which can run code hidden inside them. Only use `--host 0.0.0.0` on a trusted network. The default, `127.0.0.1`, is reachable only from the Jetson itself.

---

## 3. The interface

```
┌──────────────┬────────────────────────────────────────────────────────────┐
│ Model Lab    │ Model Lab › Compare     CUDA  TensorRT  ORT GPU  15W  Idle │
│              ├────────────────────────────────────────────────────────────┤
│ Overview     │                                                            │
│ WORKFLOW     │   page content: forms, progress, results                   │
│ 1 Convert    │                                                            │
│ 2 Inference  │                                                            │
│ 3 Compare    │                                                            │
│ ASSETS       │                                                            │
│ Models       │                                                            │
│ Videos       │                                                            │
│              │                                                            │
│ ● device     │                                                            │
│ ☾ theme      │                                                            │
└──────────────┴────────────────────────────────────────────────────────────┘
```

**Sidebar (left).** Navigation, the device name with a status dot (green = CUDA and TensorRT available), and a **light/dark theme** switch.

**Status bar (top).** Shows where you are, then CUDA, TensorRT and ONNX Runtime (GPU/CPU) status, the power mode and free disk space. On the right is the job indicator: *Idle*, or the job that is running.

| Page | What you do there |
|---|---|
| **Overview** | Summary tiles (models, TensorRT engines, test videos, saved runs, free space), the three workflow steps, a **Device** panel listing everything installed, and a table of recent runs. |
| **1 · Convert** | Upload a `.pt` (drag and drop) or pick a model from the library, then tick the output formats and set the image size. Builds ONNX FP32, TensorRT FP32 and TensorRT FP16; INT8 is shown as *on hold*. A live progress panel and console log appear under the form. |
| **2 · Inference** | Run **one** model on a video from the library, an uploaded video or image, a file path on the Jetson, a USB camera (`0`) or an RTSP stream. Shows the annotated output, speed tiles, charts and detections per class. Past runs are listed on the right. |
| **3 · Compare** | Tick several models and pick a **baseline** and a video. Each model runs on the same frames, one after another. Shows a results table, charts, synchronised videos and downloads. Saved comparisons are listed on the right. |
| **Models** | The model library: one card per model with architecture, parameters, GFLOPs, training size and class names, and a table of its formats (size, input size, where and when it was built). Buttons let you **Run** a format, download it, delete it, or **Add formats**. **Import an existing .onnx / .engine** is at the top. |
| **Videos** | Upload test videos once and reuse them, so every model sees exactly the same frames. Shows resolution, FPS, frame count and duration. |

Every result has its own link, such as `http://<jetson-ip>:8000/#compare/<run-id>`. Bookmark it or send it to the student, and it opens straight to that result.

---

## 4. Workflow for the research comparison

1. **Videos:** upload the test video, for example `car_fall.mp4`.
2. **Convert:** drop `yolov8_custom.pt` in the upload box and name it, for example `yolov8s_carfall`. Set the **image size** the model was trained with. Keep ONNX, TensorRT FP32 and TensorRT FP16 ticked, then click **Start conversion**. Each engine takes a few minutes on Orin Nano. Repeat for the YOLO11 model, e.g. `yolo11s_carfall`. Jobs queue up, so you can submit both straight away.
3. **Models:** check that both cards list four formats: PyTorch, ONNX, TensorRT FP32 and TensorRT FP16.
4. **Inference (optional):** do a quick run with one model to check that the detections look right.
5. **Compare:**
   - Give the comparison a title, e.g. *YOLOv8s vs YOLO11s · car_fall.mp4*.
   - Tick all 8 models (2 architectures × 4 formats). **Select all** ticks every format of a model.
   - Choose the **baseline**. By default it is the first *PyTorch FP32* ticked.
   - Pick the video. Optionally open **Accuracy with labels** and enter a dataset YAML to also get mAP.
   - Add **notes** (power mode, camera, anything the mentor should know) and click **Run comparison**.
6. **Read the result:** summary tiles, results table, charts, videos (**Play all** starts them in sync) and the test setup.
7. **Hand in:** download **summary.csv** and **All data .zip**, or **Data + videos .zip**, and send the result link.

**For a fair YOLOv8 vs YOLO11 comparison:** convert both at the same image size, run them in one comparison on the same video, and keep the same confidence, IoU and warm-up settings.

### Run settings

| Setting | Default | Meaning |
|---|---|---|
| Image size | auto | Only changeable for `.pt`. Exported models have a fixed input size, shown next to every model. |
| Confidence | 0.25 | Minimum score for a box to count |
| IoU (NMS) | 0.7 | Overlap threshold for removing duplicate boxes |
| Device | Auto | GPU if available. TensorRT engines always use the GPU. |
| Warm-up frames | 10 | Processed but left out of the statistics: the first frames include start-up time |
| Max frames | 0 | Stop after this many frames (0 = whole video). **Required** for cameras and streams. |
| Save annotated output | on | Writes the output video. Lowers *pipeline* FPS, not *model* FPS. |

---

## 5. What is measured

| Metric | Meaning |
|---|---|
| **Inference p50 / p95 ms** | Pure model time per frame (median / 95th percentile) |
| **Latency ms** | Pre-process + inference + post-process (NMS), mean ± standard deviation |
| **Model FPS** | `1000 ÷ latency`. **The main speed number for comparing formats.** |
| **Pipeline FPS** | Real loop speed, including reading frames, drawing boxes and writing the video |
| **Speed-up** | Baseline latency ÷ this model's latency |
| **Det / frame** | Average number of boxes per frame |
| **Agreement F1, Mean IoU** | How closely each model's boxes match the **baseline's** on every frame (same class, IoU ≥ 0.5). 1.000 means identical output. It shows how much a conversion changed the predictions **without needing labels**, but it is not accuracy against ground truth. |
| **Power W, FPS/W** | Board input power from `tegrastats` during the run, and energy efficiency |
| **GPU %, GPU °C max** | Average GPU load and peak temperature |
| **mAP50, mAP50-95** | Accuracy on a labelled dataset. Only shown when a dataset YAML was given. |

In the results table, **green cells** mark the best value in each column. The summary tiles above the table show the fastest model, its FPS and speed-up, the lowest agreement score and the number of measured frames.

**Charts:** model FPS, inference time, speed-up, agreement, power, FPS per watt and mAP as bar charts. Latency per frame, detections per frame, power over time and GPU temperature as line charts. Power and temperature charts only appear when `tegrastats` is available.

---

## 6. Saved data

Everything is stored under `data/` next to `app.py`:

```
data/
├── env.json                    device and software versions
├── models/<model-name>/        one folder per model
│   ├── model.json                architecture, task, classes, params, GFLOPs
│   ├── pt_fp32.pt   + .json      original weights
│   ├── onnx_fp32.onnx + .json
│   ├── trt_fp32.engine + .json   .json = input size, size, build time, device, TensorRT version
│   └── trt_fp16.engine + .json
├── videos/                     test-video library
└── jobs/<run-id>/              one folder per conversion / inference / comparison
    ├── job.json                  settings, models, status
    ├── log.txt                   full console output
    ├── comparison.json           all results + chart data + environment info
    ├── summary.csv               one row per model (open in Excel)
    └── results/<n>/              one folder per model:
        ├── summary.json            all statistics for this model
        ├── frames.csv              one row per frame: times, detections, count per class
        ├── detections.jsonl        every box on every frame (class, confidence, x1 y1 x2 y2)
        ├── system.csv              power / GPU / temperature / RAM every 0.5 s
        ├── output.mp4              annotated video (model name + live FPS in the corner)
        └── val/, val.json          mAP results (if a dataset YAML was given)
```

**All data .zip** downloads a run without its videos. **Data + videos .zip** includes them. The **Test setup** section of each result records the device, L4T/JetPack, power mode, CUDA/TensorRT/PyTorch/Ultralytics/ONNX Runtime versions, source video, settings and notes, so the numbers can be reproduced.

---

## 7. How it works and files

The browser sends a request, and `app.py` puts a job in a queue. **One job runs at a time**, because the Jetson has one GPU and limited memory. Each step runs `worker.py` in its own process, so GPU memory is freed afterwards and a crash cannot stop the web server. Job state is saved to disk: after a restart, the history is still there, queued jobs continue, and a job that was running is marked as interrupted. Running and queued jobs can be cancelled from their progress panel.

| File | Purpose |
|---|---|
| `app.py` | Web server: pages, API, model and video libraries, job queue |
| `worker.py` | Does the heavy work in a separate process: `env`, `convert`, `infer`, `val` |
| `analysis.py` | Builds the comparison: agreement scores, chart data, `summary.csv` |
| `static/index.html`, `app.js`, `style.css` | The web interface (plain HTML/CSS/JS, no build step) |
| `modelopt.py`, `run_video.py` | Older stand-alone command-line tools (still work) |
| `data/` | Created on first start (see [Saved data](#6-saved-data)) |

### Command-line use of the worker

```bash
python3 worker.py infer --model data/models/yolov8s_carfall/trt_fp16.engine --source test.mp4 --out /tmp/run1 --imgsz 640 --save
python3 worker.py val   --model data/models/yolov8s_carfall/trt_fp16.engine --data data.yaml --out /tmp/run1 --imgsz 640
python3 worker.py env   --out /tmp/env.json
```

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Status bar shows **No CUDA** / **No TensorRT** | You are not on the Jetson, or PyTorch is not NVIDIA's Jetson build. Only ONNX conversion works without them. |
| Engine build fails with an out-of-memory error | Close other programs (and the desktop GUI), then convert one format at a time. |
| Imported `.engine` fails to load | It was built on another device or TensorRT version. Rebuild it here from the `.pt`. |
| ONNX is very slow, **ORT CPU** tag | Install the Jetson GPU build of ONNX Runtime (see [install](#1-install-on-the-jetson-orin-nano)). |
| Video shows "mp4v, browsers cannot play" | `sudo apt install ffmpeg`, then run again. You can still download the file. |
| No power / GPU readings | `tegrastats` was not found. Readings fall back to CPU and RAM only. |
| Detections look wrong after conversion | Check that the image size matches training. Exported models have a fixed input size, shown next to each model. |
| "A different model is already called …" | Model names must be unique. Pick a new name, or delete the old model on the **Models** page. |
| "All selected formats already exist" | Tick **Rebuild formats that already exist**, or choose a different image size. |
| Camera / RTSP run is refused | Set **Max frames**: live sources have no end. |
| Page looks old after an update | Hard-refresh the browser with **Ctrl+Shift+R**. |
| Jetson storage full | Delete old runs (🗑 in the run lists) and unused videos and models. Free space is shown in the status bar and on the Overview page. |

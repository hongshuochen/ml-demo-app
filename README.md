# TFLite Live Camera — Classify / Detect / Off

A native Android (Kotlin) app that runs an on-device TensorFlow Lite model on the
live camera stream with **CameraX**, and lets you switch models from a small
on-screen selector. Three modes:

- **Classify** — the quantization-aware **uint8** multi-task classifier
  (Task 1 `Pinch`/`Other`, Task 2 `Human`/`Other`).
- **Detect** — a **generic YOLO** TFLite detector; boxes are drawn on the preview.
  Ships configured for a 1-class **hand** detector, but works with other YOLO
  exports (see below).
- **Off** — camera preview only, no inference.

Only one model runs at a time. Switching stops the current model and loads the
selected one **on the analysis thread**, so it never races with an in-flight frame.

## Architecture

| File | Responsibility |
|------|----------------|
| `MainActivity.kt` | CameraX preview + `ImageAnalysis`, permissions, the segmented mode selector, model switching, result dispatch |
| `Vision.kt` | `VisionMode`, `VisionModel` interface, `VisionResult` (sealed), `DetectionBox` |
| `FrameUtils.kt` | `loadMappedModel()` + reusable `ImagePreprocessor` (frame → upright → resized RGB) |
| `ClassificationModel.kt` | The uint8 classifier, implementing `VisionModel` |
| `YoloDetector.kt` | Generic YOLO decoder (`YoloConfig`) implementing `VisionModel` |
| `OverlayView.kt` | Custom view that draws detection boxes (FILL_CENTER mapping) |

## Add your models

Drop the files into `app/src/main/assets/`:

```
app/src/main/assets/model.tflite   # classification  ([1,320,320,3] uint8 -> [1,4] uint8)
app/src/main/assets/yolo.tflite    # YOLO detection
```

If a file is missing, the app still runs the camera and shows a clear message when
you pick that mode.

### Classification model contract

| | |
|---|---|
| Input | `[1, 320, 320, 3]` uint8, pixels **0–255**, no normalization |
| Output | `[1, 4]` uint8 — `[pinch_other, pinch, human_other, human]` |
| Task 1 | `Pinch` if `output[1] > output[0]`, else `Other` |
| Task 2 | `Human` if `output[3] > output[2]`, else `Other` |

### Generic YOLO support

`YoloDetector` reads input size & dtype from the model and auto-detects the output
format. Supported single-output layouts:

- **Grid** (v5/v8/v11) — `[1, 4+nc, N]` / `[1, N, 4+nc]` (no objectness) and
  `[1, 5+nc, N]` / `[1, N, 5+nc]` (with objectness). `xywh`-centre boxes; NMS in-app.
- **End-to-end** (YOLOv10 / **YOLO26 / yolo26n**) — `[1, N, 6]` =
  `[x1, y1, x2, y2, score, class_id]`, `xyxy`, already NMS-free.

Boxes may be normalized 0–1 or input-pixel units (auto-detected). Input may be
float32 (scaled to 0–1) or uint8 (raw 0–255); output may be float32 or quantised
(dequantised via the tensor's params).

To use a **different** YOLO model, change the defaults in `YoloConfig` (or pass a
custom one when constructing `YoloDetector` in `MainActivity`):

```kotlin
YoloConfig(
    modelAsset = "yolo.tflite",
    labels = listOf("hand"),               // class names, in model index order
    confidenceThreshold = 0.35f,
    iouThreshold = 0.45f,
    // outputFormat defaults to AUTO; for production pin it explicitly:
    outputFormat = YoloOutputFormat.END_TO_END,   // for yolo26n / v10
    // coordsArePixels = true,              // if AUTO mis-detects coord units
)
```

> **YOLO26 note:** a default YOLO26 TFLite export is end-to-end (`[1, 300, 6]`,
> NMS-free). AUTO detects this, but pinning `outputFormat = END_TO_END` is the
> robust choice for deployment. (An Edge TPU export, by contrast, falls back to the
> grid format.)

## Performance

- Inference runs on a single background thread (also used for model load/release).
- `STRATEGY_KEEP_ONLY_LATEST` + a `compareAndSet` busy-guard → frames are skipped
  while a previous inference is still running.
- Every `ImageProxy` is closed in a `finally`.
- Input/output `ByteBuffer`s, the resize bitmap, and pixel arrays are reused.

## Build & run

**Android Studio:** open the folder, let Gradle sync, pick a device, press **Run**.

**Command line:**

```bash
./gradlew assembleDebug     # debug APK -> app/build/outputs/apk/debug/app-debug.apk
./gradlew installDebug      # install onto a connected device/emulator
```

## Requirements

`minSdk 26`, `compileSdk 35` · CameraX `1.4.1` · TensorFlow Lite `2.16.1` ·
Kotlin `2.0.21` · AGP `8.7.3`

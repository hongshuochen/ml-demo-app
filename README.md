# TFLite Live Camera Classifier

A native Android (Kotlin) app that runs a quantization-aware **uint8** TensorFlow
Lite model on the live camera stream using **CameraX** + the **TFLite Interpreter**.

For every frame it predicts two tasks and shows them on screen:

- **Task 1** — `Pinch` vs `Other`  (`output[1] > output[0]` → Pinch)
- **Task 2** — `Human` vs `Other`  (`output[3] > output[2]` → Human)

## Model contract

| | |
|---|---|
| Input shape | `[1, 320, 320, 3]` |
| Input type  | `uint8`, pixel values **0–255** (no normalization) |
| Output      | `[1, 4]` `uint8`, values 0–255 |
| Output order| `[pinch_other, pinch, human_other, human]` |

## Add your model

Copy your model to:

```
app/src/main/assets/model.tflite
```

(If it has a different name, change `MODEL_ASSET` in `TfLiteClassifier.kt`.)
The app shows a clear on-screen message if the model is missing — the camera
preview still runs.

## Build & run

**Android Studio:** open this folder, let Gradle sync, plug in a device (or start
an emulator), press **Run**.

**Command line:**

```bash
# Build a debug APK
./gradlew assembleDebug

# Install onto a connected device/emulator
./gradlew installDebug
```

The debug APK lands at `app/build/outputs/apk/debug/app-debug.apk`.

## How it works

- `MainActivity.kt` — permissions, CameraX preview + `ImageAnalysis`
  (`STRATEGY_KEEP_ONLY_LATEST`, RGBA_8888 output), single-thread executor, UI.
- `TfLiteClassifier.kt` — loads the memory-mapped model, converts each frame to a
  rotated 320×320 RGB bitmap, packs it into a reused uint8 buffer, runs inference,
  and decodes the four uint8 outputs into two predictions.

### Performance notes

- Inference runs on a dedicated single background thread.
- `STRATEGY_KEEP_ONLY_LATEST` + closing each `ImageProxy` only after inference
  means exactly one frame is processed at a time (no backlog).
- Input/output `ByteBuffer`s, the 320×320 bitmap, and the pixel array are
  allocated once and reused.

## Requirements

- `minSdk 26`, `compileSdk 35`
- CameraX `1.4.1`, TensorFlow Lite `2.16.1`, Kotlin `2.0.21`, AGP `8.7.3`

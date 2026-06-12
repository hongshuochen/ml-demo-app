Place your TensorFlow Lite model(s) in THIS folder.

1) Classification model  ->  model.tflite
   Input : [1, 320, 320, 3], uint8, pixel range 0..255 (NO normalization)
   Output: [1, 4],          uint8, range 0..255
           [pinch_other, pinch, human_other, human]

2) YOLO detection model  ->  yolo.tflite   (incl. YOLO26 / yolo26n)
   Input : [1, H, W, 3], float32 (scaled to 0..1) OR uint8 (raw 0..255).
           H/W are read from the model (e.g. 320x320 or 640x640).
   Output: a single tensor, auto-detected:
             GRID (v5/v8/v11):
               [1, 4+nc, N] / [1, N, 4+nc]   (no objectness)
               [1, 5+nc, N] / [1, N, 5+nc]   (with objectness)
               boxes xywh (centre); NMS applied in-app.
             END-TO-END (YOLOv10 / YOLO26):
               [1, N, 6] = [x1, y1, x2, y2, score, class_id]
               xyxy, already NMS-free.
           Coords may be normalised 0..1 or input pixels (auto-detected).
   The current default expects ONE class: "hand".
   If AUTO guesses wrong, set YoloConfig.outputFormat (END_TO_END for yolo26n)
   and/or coordsArePixels in YoloDetector.kt.

Changing model names / classes:
  - Classification asset name: MODEL_ASSET in ClassificationModel.kt
  - YOLO asset name, labels, thresholds: YoloConfig (defaults in YoloDetector.kt;
    pass a custom YoloConfig from MainActivity to use a different YOLO model).

If a model file is missing, the app still runs the camera preview and shows a
clear message when you select that mode.

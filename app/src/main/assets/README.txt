Place your TensorFlow Lite model in THIS folder as:

    model.tflite

The app expects:
  Input : [1, 320, 320, 3], uint8, pixel range 0..255 (NO normalization)
  Output: [1, 4],          uint8, range 0..255
          [pinch_other, pinch, human_other, human]

If your file has a different name, change MODEL_ASSET in
app/src/main/java/com/example/tfliteclassifier/TfLiteClassifier.kt

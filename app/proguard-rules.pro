# TensorFlow Lite uses native/reflection-loaded classes. These rules only matter
# if you ever enable minification (isMinifyEnabled = true).
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

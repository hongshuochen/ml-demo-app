plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.tfliteclassifier"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.tfliteclassifier"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            // Keep minification off so TFLite reflection-loaded classes survive.
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    // The .tflite asset MUST stay uncompressed so it can be memory-mapped.
    androidResources {
        noCompress += "tflite"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // AndroidX core + a lightweight Activity base that gives us
    // registerForActivityResult() and LifecycleOwner (for CameraX).
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.3")

    // CameraX: core APIs, the Camera2 backend, lifecycle binding, and PreviewView.
    val cameraX = "1.4.1"
    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")

    // TensorFlow Lite interpreter (CPU). Add tensorflow-lite-gpu for a GPU delegate.
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
}

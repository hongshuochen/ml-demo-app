package com.example.tfliteclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Shows a live CameraX preview and runs the TFLite classifier on every frame.
 *
 * Threading model:
 *  - All frame analysis + inference runs on [analysisExecutor], a single thread.
 *  - CameraX is configured with STRATEGY_KEEP_ONLY_LATEST and we only close each
 *    [ImageProxy] when analysis finishes, so exactly one frame is ever in flight.
 *  - UI updates are posted back to the main thread.
 */
class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var resultText: TextView

    private lateinit var analysisExecutor: ExecutorService

    /** The model wrapper. Null if the model could not be loaded. */
    private var classifier: TfLiteClassifier? = null

    /** Runtime CAMERA permission request. */
    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                resultText.text = getString(R.string.camera_permission_required)
                Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        resultText = findViewById(R.id.resultText)

        analysisExecutor = Executors.newSingleThreadExecutor()

        // Load the model up front. If it's missing/incompatible, keep the preview
        // alive but tell the user instead of crashing.
        classifier = try {
            TfLiteClassifier(this)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model", e)
            resultText.text = getString(R.string.model_load_failed, e.message ?: "")
            null
        }

        if (isCameraPermissionGranted()) {
            startCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun isCameraPermissionGranted(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            bindUseCases(cameraProviderFuture.get())
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindUseCases(cameraProvider: ProcessCameraProvider) {
        // Live preview rendered to the PreviewView surface.
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        // Ask for a modest analysis resolution to keep inference fast; CameraX
        // falls back to the closest supported size.
        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(640, 480),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER,
                ),
            )
            .build()

        // RGBA_8888 output makes the bitmap conversion a straight copy, and
        // KEEP_ONLY_LATEST drops stale frames instead of queueing them.
        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { it.setAnalyzer(analysisExecutor, ::analyzeFrame) }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis,
            )
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }

    /**
     * Runs on [analysisExecutor] for every frame. The `finally` block guarantees
     * the proxy is closed, which is also what unblocks delivery of the next frame.
     */
    private fun analyzeFrame(imageProxy: ImageProxy) {
        val model = classifier
        if (model == null) {
            imageProxy.close()
            return
        }
        try {
            val result = model.classify(imageProxy)
            // Touch the UI only on the main thread.
            resultText.post { renderResult(result) }
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun renderResult(r: ClassificationResult) {
        val task1 = if (r.task1IsPinch) "Pinch" else "Other"
        val task2 = if (r.task2IsHuman) "Human" else "Other"
        // Confidence-like score = winning raw value scaled to a percentage.
        val task1Pct = maxOf(r.pinchOther, r.pinch) * 100 / 255
        val task2Pct = maxOf(r.humanOther, r.human) * 100 / 255

        resultText.text = String.format(
            Locale.US,
            "Task 1 (pinch): %s  (%d%%)\n" +
                "Task 2 (human): %s  (%d%%)\n\n" +
                "Raw uint8 output: [%d, %d, %d, %d]\n" +
                "  pinch_other = %-3d   pinch = %-3d\n" +
                "  human_other = %-3d   human = %-3d\n\n" +
                "Inference: %d ms",
            task1, task1Pct,
            task2, task2Pct,
            r.pinchOther, r.pinch, r.humanOther, r.human,
            r.pinchOther, r.pinch,
            r.humanOther, r.human,
            r.inferenceTimeMs,
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        // Drain the analysis thread BEFORE releasing the model. shutdown() lets the
        // in-flight frame finish; awaitTermination() blocks (briefly — inference is a
        // few ms) until it does. This guarantees interpreter.close()/bitmap.recycle()
        // never run while classify() is still using them on the analysis thread,
        // which would otherwise be a native use-after-free.
        analysisExecutor.shutdown()
        try {
            if (!analysisExecutor.awaitTermination(1, TimeUnit.SECONDS)) {
                analysisExecutor.shutdownNow()
            }
        } catch (e: InterruptedException) {
            analysisExecutor.shutdownNow()
            Thread.currentThread().interrupt()
        }
        classifier?.close()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}

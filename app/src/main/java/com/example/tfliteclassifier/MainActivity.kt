package com.example.tfliteclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Live CameraX preview with a switchable on-device vision model.
 *
 * Modes (chosen via the bottom segmented selector):
 *   - Classify : the multi-task uint8 classifier
 *   - Detect   : a generic YOLO TFLite detector (boxes drawn on the overlay)
 *   - Off      : camera preview only, no inference
 *
 * Threading / lifecycle:
 *   - One single-thread [analysisExecutor] runs BOTH frame analysis and model
 *     load/release, so switching never races with an in-flight frame and only one
 *     model is ever active.
 *   - ImageAnalysis uses STRATEGY_KEEP_ONLY_LATEST; frames are skipped while a
 *     previous inference is still running; every ImageProxy is closed.
 */
class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var statusText: TextView
    private lateinit var segClassify: TextView
    private lateinit var segDetect: TextView
    private lateinit var segOff: TextView

    private lateinit var analysisExecutor: ExecutorService

    /**
     * The active model, or null for OFF / while a switch is loading. Mutated and
     * read only on [analysisExecutor] (and in onDestroy, after it has drained).
     */
    @Volatile
    private var activeModel: VisionModel? = null

    /** Currently selected mode (main-thread UI state). */
    private var selectedMode = VisionMode.CLASSIFY

    /** True while an inference is in progress; makes "skip if busy" explicit. */
    private val isAnalyzing = AtomicBoolean(false)

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                statusText.text = getString(R.string.camera_permission_required)
                Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        statusText = findViewById(R.id.statusText)
        segClassify = findViewById(R.id.segClassify)
        segDetect = findViewById(R.id.segDetect)
        segOff = findViewById(R.id.segOff)

        segClassify.setOnClickListener { switchMode(VisionMode.CLASSIFY) }
        segDetect.setOnClickListener { switchMode(VisionMode.DETECT) }
        segOff.setOnClickListener { switchMode(VisionMode.OFF) }

        analysisExecutor = Executors.newSingleThreadExecutor()

        // Start in classification mode (preserves the original app behaviour).
        switchMode(VisionMode.CLASSIFY)

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
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({ bindUseCases(future.get()) }, ContextCompat.getMainExecutor(this))
    }

    private fun bindUseCases(cameraProvider: ProcessCameraProvider) {
        // Preview and analysis share a 4:3 aspect ratio so the detection overlay
        // maps cleanly onto what the PreviewView displays.
        val selector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .build()

        val preview = Preview.Builder()
            .setResolutionSelector(selector)
            .build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }

        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(selector)
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
     * Runs on [analysisExecutor] for every delivered frame. The `finally` block
     * always closes the proxy (which also unblocks the next frame).
     */
    private fun analyzeFrame(imageProxy: ImageProxy) {
        val model = activeModel
        if (model == null) {
            // OFF, or a switch is still loading: just drop the frame.
            imageProxy.close()
            return
        }
        if (!isAnalyzing.compareAndSet(false, true)) {
            // A previous inference is still running: skip this frame.
            imageProxy.close()
            return
        }
        try {
            val result = model.analyze(imageProxy)
            runOnUiThread { renderResult(result) }
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
        } finally {
            isAnalyzing.set(false)
            imageProxy.close()
        }
    }

    /**
     * Switches the active model. The old model is closed and the new one loaded on
     * [analysisExecutor], so the swap is serialised against frame analysis and the
     * previous model is always fully stopped first.
     */
    private fun switchMode(mode: VisionMode) {
        selectedMode = mode
        updateSelectorUi(mode)
        overlayView.clear()
        statusText.text = if (mode == VisionMode.OFF) {
            getString(R.string.status_off)
        } else {
            getString(R.string.loading_mode, mode.label)
        }

        analysisExecutor.execute {
            activeModel?.close()
            activeModel = null
            if (mode == VisionMode.OFF) return@execute
            try {
                val model: VisionModel = when (mode) {
                    VisionMode.CLASSIFY -> ClassificationModel(applicationContext)
                    VisionMode.DETECT -> YoloDetector(applicationContext)
                    VisionMode.OFF -> return@execute
                }
                activeModel = model
                runOnUiThread {
                    // Ignore a stale load if the user switched again meanwhile.
                    if (selectedMode == mode) {
                        statusText.text = getString(R.string.model_ready, model.displayName)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model for $mode", e)
                runOnUiThread {
                    if (selectedMode == mode) {
                        statusText.text = getString(R.string.model_load_failed, e.message ?: "")
                    }
                }
            }
        }
    }

    private fun updateSelectorUi(mode: VisionMode) {
        val segments = listOf(
            VisionMode.CLASSIFY to segClassify,
            VisionMode.DETECT to segDetect,
            VisionMode.OFF to segOff,
        )
        for ((m, view) in segments) {
            val selected = m == mode
            view.setBackgroundResource(if (selected) R.drawable.bg_segment_selected else 0)
            view.setTextColor(if (selected) Color.WHITE else SEGMENT_INACTIVE)
            view.typeface = if (selected) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        }
    }

    private fun renderResult(result: VisionResult) {
        // A result computed for the previous mode can land here after the user has
        // already switched (runOnUiThread posts to the main queue). Drop stale
        // results so they can't repaint the overlay/status for the wrong mode.
        when (result) {
            is VisionResult.Classification -> {
                if (selectedMode != VisionMode.CLASSIFY) return
                overlayView.clear()
                statusText.text = formatClassification(result)
            }
            is VisionResult.Detection -> {
                if (selectedMode != VisionMode.DETECT) {
                    overlayView.clear()
                    return
                }
                overlayView.setDetections(result.boxes, result.frameWidth, result.frameHeight)
                statusText.text = String.format(
                    Locale.US,
                    "%s\n%d detection(s)  •  %d ms",
                    activeModel?.displayName ?: "Detector",
                    result.boxes.size,
                    result.inferenceTimeMs,
                )
            }
        }
    }

    private fun formatClassification(r: VisionResult.Classification): String {
        val task1 = if (r.task1IsPinch) "Pinch" else "Other"
        val task2 = if (r.task2IsHuman) "Human" else "Other"
        val task1Pct = maxOf(r.pinchOther, r.pinch) * 100 / 255
        val task2Pct = maxOf(r.humanOther, r.human) * 100 / 255
        return String.format(
            Locale.US,
            "Task 1 (pinch): %s  (%d%%)\n" +
                "Task 2 (human): %s  (%d%%)\n" +
                "raw [%d, %d, %d, %d]  •  %d ms",
            task1, task1Pct,
            task2, task2Pct,
            r.pinchOther, r.pinch, r.humanOther, r.human,
            r.inferenceTimeMs,
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        if (!::analysisExecutor.isInitialized) return
        // Release the model ON the analysis thread so close() is ordered strictly
        // after any in-flight analyze() — never concurrent with native
        // interpreter.run() (a use-after-free). We deliberately do NOT close on the
        // main thread: if awaitTermination times out, shutdownNow() can't interrupt
        // a native call, so we'd rather leak than free under a live call.
        analysisExecutor.execute {
            activeModel?.close()
            activeModel = null
        }
        analysisExecutor.shutdown()
        try {
            if (!analysisExecutor.awaitTermination(2, TimeUnit.SECONDS)) {
                analysisExecutor.shutdownNow()
            }
        } catch (e: InterruptedException) {
            analysisExecutor.shutdownNow()
            Thread.currentThread().interrupt()
        }
    }

    companion object {
        private const val TAG = "MainActivity"
        private val SEGMENT_INACTIVE = Color.parseColor("#B3FFFFFF")
    }
}

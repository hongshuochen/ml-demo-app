package com.example.tfliteclassifier

import androidx.camera.core.ImageProxy

/** The three things the user can select in the on-screen mode picker. */
enum class VisionMode(val label: String) {
    CLASSIFY("Classify"),
    DETECT("Detect"),
    OFF("Off"),
}

/**
 * A model that turns one CameraX frame into a [VisionResult].
 *
 * Implementations reuse internal buffers and are therefore **single-threaded**:
 * [analyze] must only be called from the CameraX analysis executor, and [close]
 * only after that executor has drained.
 *
 * Implementations must NOT close the [ImageProxy] — the caller owns it.
 */
interface VisionModel {
    /** Human-readable name shown in the status overlay. */
    val displayName: String

    fun analyze(image: ImageProxy): VisionResult

    /** Releases the interpreter and any native buffers/bitmaps. */
    fun close()
}

/** Result of running one model on one frame. */
sealed interface VisionResult {
    val inferenceTimeMs: Long

    /** Multi-task classifier output (pinch/other, human/other). */
    data class Classification(
        val pinchOther: Int,
        val pinch: Int,
        val humanOther: Int,
        val human: Int,
        val task1IsPinch: Boolean,
        val task2IsHuman: Boolean,
        override val inferenceTimeMs: Long,
    ) : VisionResult

    /**
     * Detector output. [boxes] are normalised (0..1) against an upright frame of
     * [frameWidth] x [frameHeight], which lets the overlay map them onto the
     * preview regardless of the analysis resolution.
     */
    data class Detection(
        val boxes: List<DetectionBox>,
        val frameWidth: Int,
        val frameHeight: Int,
        override val inferenceTimeMs: Long,
    ) : VisionResult
}

/** One detected object. Edges are normalised (0..1) against the upright frame. */
data class DetectionBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val label: String,
    val score: Float,
)

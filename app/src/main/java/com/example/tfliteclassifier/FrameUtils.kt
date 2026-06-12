package com.example.tfliteclassifier

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Rect
import androidx.camera.core.ImageProxy
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Memory-maps an uncompressed `.tflite` asset so the weights are never copied
 * onto the Java heap. Requires `androidResources { noCompress += "tflite" }`.
 */
fun loadMappedModel(context: Context, assetName: String): MappedByteBuffer {
    context.assets.openFd(assetName).use { fd ->
        FileInputStream(fd.fileDescriptor).use { input ->
            return input.channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength,
            )
        }
    }
}

/**
 * Reusable converter: CameraX [ImageProxy] -> upright, stretched RGB bitmap whose
 * pixels are exposed as ARGB ints. One instance per model (input sizes differ).
 *
 * NOT thread-safe — it reuses a bitmap/canvas/scratch buffers, so call [process]
 * from a single thread (the analysis executor).
 */
class ImagePreprocessor(private val inputWidth: Int, private val inputHeight: Int) {

    /** ARGB ints of the most recent resized frame (length = inputWidth*inputHeight). */
    val pixels = IntArray(inputWidth * inputHeight)

    /** Width of the upright (rotated) frame the resize was taken from. */
    var uprightWidth = 0
        private set

    /** Height of the upright (rotated) frame the resize was taken from. */
    var uprightHeight = 0
        private set

    private val resizedBitmap =
        Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(resizedBitmap)
    private val srcRect = Rect()
    private val dstRect = Rect(0, 0, inputWidth, inputHeight)
    private val rotateMatrix = Matrix()

    fun process(imageProxy: ImageProxy) {
        // RGBA_8888 analysis output -> toBitmap() is a straight copy (CameraX
        // strips any row padding for us).
        val frame: Bitmap = imageProxy.toBitmap()

        // The two per-frame bitmaps are native; recycle them in finally so a throw
        // (e.g. OOM from createBitmap) can't leak them on the hot path.
        var upright: Bitmap? = null
        try {
            // Rotate upright using the per-frame sensor rotation.
            val rotation = imageProxy.imageInfo.rotationDegrees
            upright = if (rotation == 0) {
                frame
            } else {
                rotateMatrix.reset()
                rotateMatrix.postRotate(rotation.toFloat())
                Bitmap.createBitmap(frame, 0, 0, frame.width, frame.height, rotateMatrix, true)
            }

            uprightWidth = upright.width
            uprightHeight = upright.height

            // Stretch-to-fit into the reused destination bitmap.
            srcRect.set(0, 0, upright.width, upright.height)
            canvas.drawBitmap(upright, srcRect, dstRect, null)

            resizedBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        } finally {
            // Release the per-frame bitmaps (toBitmap()/createBitmap() allocate fresh).
            if (upright != null && upright !== frame) upright.recycle()
            frame.recycle()
        }
    }

    fun close() = resizedBitmap.recycle()
}

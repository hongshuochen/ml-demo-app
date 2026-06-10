package com.example.tfliteclassifier

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Immutable, thread-safe snapshot of one frame's prediction. Created on the
 * analysis thread and handed to the UI thread for display.
 *
 * The four raw values map to the model's concatenated multi-task output:
 *   index 0 -> [pinchOther]  Task 1 "Other"
 *   index 1 -> [pinch]       Task 1 "Pinch"
 *   index 2 -> [humanOther]  Task 2 "Other"
 *   index 3 -> [human]       Task 2 "Human"
 */
data class ClassificationResult(
    val pinchOther: Int,
    val pinch: Int,
    val humanOther: Int,
    val human: Int,
    val task1IsPinch: Boolean,
    val task2IsHuman: Boolean,
    val inferenceTimeMs: Long,
)

/**
 * Loads a quantization-aware (uint8) TensorFlow Lite model from `assets/` and
 * runs it on CameraX frames.
 *
 *  - Input  : `[1, 320, 320, 3]` uint8. Pixel values are kept in 0..255 — they
 *             are fed to the interpreter **without** any normalization, mean
 *             subtraction, or scaling. The model's own quantization parameters
 *             handle the mapping internally.
 *  - Output : `[1, 4]` uint8, values 0..255 =
 *             `[pinch_other, pinch, human_other, human]`.
 *
 * NOT thread-safe: it reuses internal scratch buffers, so [classify] must be
 * called from a single thread (the CameraX analysis executor).
 */
class TfLiteClassifier(context: Context) {

    private val interpreter: Interpreter

    // --- Reusable scratch buffers (allocated once, reused every frame) ---------

    /** uint8 input tensor: 320 * 320 * 3 bytes, one byte per channel value. */
    private val inputBuffer: ByteBuffer

    /** uint8 output tensor: 4 bytes. */
    private val outputBuffer: ByteBuffer

    /** Destination of the 320x320 resize; pixels are read out of this. */
    private val resizedBitmap: Bitmap =
        Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val resizeCanvas: Canvas = Canvas(resizedBitmap)

    /** Holds 320*320 ARGB ints read back from [resizedBitmap]. */
    private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)

    /** Scratch RGB bytes (320*320*3): filled per frame, then bulk-copied to [inputBuffer]. */
    private val inputBytes = ByteArray(INPUT_SIZE * INPUT_SIZE * PIXEL_CHANNELS)

    private val srcRect = Rect()
    private val dstRect = Rect(0, 0, INPUT_SIZE, INPUT_SIZE)
    private val rotateMatrix = Matrix()

    init {
        val options = Interpreter.Options().apply { numThreads = NUM_THREADS }
        interpreter = Interpreter(loadModelFile(context, MODEL_ASSET), options)

        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        Log.i(
            TAG,
            "Model input : shape=${inputTensor.shape().contentToString()} " +
                "type=${inputTensor.dataType()}",
        )
        Log.i(
            TAG,
            "Model output: shape=${outputTensor.shape().contentToString()} " +
                "type=${outputTensor.dataType()}",
        )

        // uint8 == 1 byte per value. Direct, native-order buffers avoid copies.
        inputBuffer = ByteBuffer
            .allocateDirect(INPUT_SIZE * INPUT_SIZE * PIXEL_CHANNELS)
            .order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer
            .allocateDirect(NUM_OUTPUTS)
            .order(ByteOrder.nativeOrder())

        // Fail fast (and clearly) if the model doesn't match our assumptions.
        require(inputTensor.numBytes() == inputBuffer.capacity()) {
            "Model input is ${inputTensor.numBytes()} bytes, expected " +
                "${inputBuffer.capacity()} ([1,$INPUT_SIZE,$INPUT_SIZE,$PIXEL_CHANNELS] uint8)."
        }
        require(outputTensor.numBytes() == outputBuffer.capacity()) {
            "Model output is ${outputTensor.numBytes()} bytes, expected " +
                "${outputBuffer.capacity()} ([1,$NUM_OUTPUTS] uint8)."
        }
        // uint8 and int8 both occupy 1 byte, so the size checks above can't tell them
        // apart. Assert the type explicitly: an int8 model would otherwise be packed
        // and decoded as unsigned, silently corrupting inputs and inverting predictions.
        require(inputTensor.dataType() == DataType.UINT8) {
            "Model input type is ${inputTensor.dataType()}, expected UINT8."
        }
        require(outputTensor.dataType() == DataType.UINT8) {
            "Model output type is ${outputTensor.dataType()}, expected UINT8."
        }
    }

    /**
     * Full per-frame pipeline: convert -> rotate -> resize -> pack -> infer ->
     * decode. The caller still owns [imageProxy] and must close it.
     */
    fun classify(imageProxy: ImageProxy): ClassificationResult {
        val startNs = System.nanoTime()

        // 1. Get the frame as an ARGB_8888 bitmap. Because the analyzer is
        //    configured for RGBA_8888 output, this is a direct copy and CameraX
        //    transparently strips any row padding for us.
        val frame: Bitmap = imageProxy.toBitmap()

        // 2. Rotate to the display's upright orientation using the rotation the
        //    camera reports for this frame.
        val rotation = imageProxy.imageInfo.rotationDegrees
        val upright: Bitmap = if (rotation == 0) {
            frame
        } else {
            rotateMatrix.reset()
            rotateMatrix.postRotate(rotation.toFloat())
            Bitmap.createBitmap(frame, 0, 0, frame.width, frame.height, rotateMatrix, true)
        }

        // 3. Resize to 320x320 by drawing into the reused bitmap (stretch-to-fit).
        srcRect.set(0, 0, upright.width, upright.height)
        resizeCanvas.drawBitmap(upright, srcRect, dstRect, null)

        // Release the per-frame bitmaps (toBitmap()/createBitmap() allocate new ones).
        if (upright !== frame) upright.recycle()
        frame.recycle()

        // 4. Pack RGB bytes into the uint8 input tensor [1,320,320,3], 0..255.
        //    getPixels() returns ARGB ints (0xAARRGGBB); we drop alpha.
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        var i = 0
        for (pixel in pixels) {
            inputBytes[i++] = ((pixel shr 16) and 0xFF).toByte() // R
            inputBytes[i++] = ((pixel shr 8) and 0xFF).toByte()  // G
            inputBytes[i++] = (pixel and 0xFF).toByte()          // B
        }
        // One bulk native copy instead of ~307k individual put() calls.
        inputBuffer.rewind()
        inputBuffer.put(inputBytes)
        inputBuffer.rewind()

        // 5. Run inference.
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)

        // 6. Decode the 4 uint8 outputs. `and 0xFF` reads each byte as unsigned.
        val pinchOther = outputBuffer.get(0).toInt() and 0xFF
        val pinch = outputBuffer.get(1).toInt() and 0xFF
        val humanOther = outputBuffer.get(2).toInt() and 0xFF
        val human = outputBuffer.get(3).toInt() and 0xFF

        val elapsedMs = (System.nanoTime() - startNs) / 1_000_000L

        return ClassificationResult(
            pinchOther = pinchOther,
            pinch = pinch,
            humanOther = humanOther,
            human = human,
            // Task 1: Pinch if output[1] > output[0], else Other.
            task1IsPinch = pinch > pinchOther,
            // Task 2: Human if output[3] > output[2], else Other.
            task2IsHuman = human > humanOther,
            inferenceTimeMs = elapsedMs,
        )
    }

    /** Releases native resources. Call from [android.app.Activity.onDestroy]. */
    fun close() {
        interpreter.close()
        resizedBitmap.recycle()
    }

    /**
     * Memory-maps the (uncompressed) .tflite asset so the weights are never
     * copied onto the Java heap.
     */
    private fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
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

    companion object {
        private const val TAG = "TfLiteClassifier"

        /** Asset file name; change this if your model is named differently. */
        private const val MODEL_ASSET = "model.tflite"

        private const val INPUT_SIZE = 320
        private const val PIXEL_CHANNELS = 3
        private const val NUM_OUTPUTS = 4
        private const val NUM_THREADS = 4
    }
}

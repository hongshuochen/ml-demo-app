package com.example.tfliteclassifier

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The original multi-task, quantization-aware **uint8** classifier.
 *
 *  - Input  : `[1,320,320,3]` uint8, pixels kept 0..255 (NO normalization).
 *  - Output : `[1,4]` uint8 = `[pinch_other, pinch, human_other, human]`.
 *
 * Prediction logic: Task 1 = Pinch if `out[1] > out[0]`, Task 2 = Human if
 * `out[3] > out[2]`.
 */
class ClassificationModel(context: Context) : VisionModel {

    override val displayName = "Pinch / Human classifier"

    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val preprocessor = ImagePreprocessor(INPUT_SIZE, INPUT_SIZE)
    private val inputBytes = ByteArray(INPUT_SIZE * INPUT_SIZE * CHANNELS)

    init {
        interpreter = Interpreter(
            loadMappedModel(context, MODEL_ASSET),
            Interpreter.Options().apply { numThreads = NUM_THREADS },
        )

        val inT = interpreter.getInputTensor(0)
        val outT = interpreter.getOutputTensor(0)
        Log.i(
            TAG,
            "input=${inT.shape().contentToString()} ${inT.dataType()} " +
                "output=${outT.shape().contentToString()} ${outT.dataType()}",
        )

        inputBuffer = ByteBuffer
            .allocateDirect(INPUT_SIZE * INPUT_SIZE * CHANNELS)
            .order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer
            .allocateDirect(NUM_OUTPUTS)
            .order(ByteOrder.nativeOrder())

        require(inT.numBytes() == inputBuffer.capacity()) {
            "Model input is ${inT.numBytes()} bytes, expected ${inputBuffer.capacity()} " +
                "([1,$INPUT_SIZE,$INPUT_SIZE,$CHANNELS] uint8)."
        }
        require(outT.numBytes() == outputBuffer.capacity()) {
            "Model output is ${outT.numBytes()} bytes, expected ${outputBuffer.capacity()} " +
                "([1,$NUM_OUTPUTS] uint8)."
        }
        // uint8 and int8 are both 1 byte; assert the type so an int8 model can't be
        // silently decoded as unsigned.
        require(inT.dataType() == DataType.UINT8) {
            "Model input type is ${inT.dataType()}, expected UINT8."
        }
        require(outT.dataType() == DataType.UINT8) {
            "Model output type is ${outT.dataType()}, expected UINT8."
        }
    }

    override fun analyze(image: ImageProxy): VisionResult {
        val start = System.nanoTime()
        preprocessor.process(image)

        // Pack RGB bytes (0..255, no normalization) into the uint8 input tensor.
        val pixels = preprocessor.pixels
        var i = 0
        for (p in pixels) {
            inputBytes[i++] = ((p shr 16) and 0xFF).toByte() // R
            inputBytes[i++] = ((p shr 8) and 0xFF).toByte()  // G
            inputBytes[i++] = (p and 0xFF).toByte()          // B
        }
        inputBuffer.rewind()
        inputBuffer.put(inputBytes)
        inputBuffer.rewind()

        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)

        // Read the 4 uint8 outputs as unsigned.
        val pinchOther = outputBuffer.get(0).toInt() and 0xFF
        val pinch = outputBuffer.get(1).toInt() and 0xFF
        val humanOther = outputBuffer.get(2).toInt() and 0xFF
        val human = outputBuffer.get(3).toInt() and 0xFF

        return VisionResult.Classification(
            pinchOther = pinchOther,
            pinch = pinch,
            humanOther = humanOther,
            human = human,
            task1IsPinch = pinch > pinchOther,
            task2IsHuman = human > humanOther,
            inferenceTimeMs = (System.nanoTime() - start) / 1_000_000L,
        )
    }

    override fun close() {
        interpreter.close()
        preprocessor.close()
    }

    companion object {
        private const val TAG = "ClassificationModel"
        private const val MODEL_ASSET = "model.tflite"
        private const val INPUT_SIZE = 320
        private const val CHANNELS = 3
        private const val NUM_OUTPUTS = 4
        private const val NUM_THREADS = 4
    }
}

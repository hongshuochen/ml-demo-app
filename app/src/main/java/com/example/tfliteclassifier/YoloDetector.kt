package com.example.tfliteclassifier

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.roundToInt

/** How to interpret the model's output tensor. */
enum class YoloOutputFormat {
    /** Decide GRID vs END_TO_END automatically from the output shape. */
    AUTO,

    /**
     * Classic raw grid (v5/v8/v11): `[1,4+nc,N]` / `[1,N,4+nc]` (no objectness) or
     * `[1,5+nc,N]` / `[1,N,5+nc]` (with objectness). Boxes are xywh-centre with
     * per-class scores; this class runs NMS.
     */
    GRID,

    /**
     * NMS-free end-to-end head (YOLOv10 / **YOLO26**): `[1,N,6]` where each row is
     * `[x1, y1, x2, y2, score, class_id]` (xyxy, already filtered). No NMS needed.
     */
    END_TO_END,
}

/** Knobs that let one decoder serve different YOLO TFLite exports. */
data class YoloConfig(
    val modelAsset: String = "yolo.tflite",
    val displayName: String = "Hand detector (YOLO)",
    /** Class names, in model index order. */
    val labels: List<String> = listOf("hand"),
    val confidenceThreshold: Float = 0.35f,
    val iouThreshold: Float = 0.45f,
    val maxDetections: Int = 25,
    val numThreads: Int = 4,
    /**
     * Output decoding strategy. AUTO handles both grid and end-to-end (YOLO26), but
     * is ambiguous for nc==1/nc==2 grids at very small input sizes (<=192 px) — for
     * production prefer setting this explicitly (END_TO_END for yolo26n).
     */
    val outputFormat: YoloOutputFormat = YoloOutputFormat.AUTO,
    /**
     * Whether box coords are in input pixels (true) or normalised 0..1 (false).
     * null = auto-detect per frame (a heuristic that works for both in practice).
     */
    val coordsArePixels: Boolean? = null,
)

/**
 * Generic single-output YOLO detector for TFLite. Handles both output styles:
 *
 *  - **Grid** (v5/v8/v11): `[1,4+nc,N]` / `[1,N,4+nc]` (v8, no objectness) and
 *    `[1,5+nc,N]` / `[1,N,5+nc]` (v5, with objectness). xywh-centre boxes +
 *    per-class scores; NMS applied here.
 *  - **End-to-end** (YOLOv10 / **YOLO26**): `[1,N,6]` = `[x1,y1,x2,y2,score,
 *    class_id]`, xyxy, already filtered; no NMS.
 *
 * The format is auto-detected (override with [YoloConfig.outputFormat]). Boxes may
 * be normalised (0..1) or in input pixels (auto-detected). Input may be float32
 * (scaled to 0..1) or uint8 (raw 0..255); output may be float32 or quantised
 * (dequantised via tensor params). Returns boxes normalised to the upright frame.
 */
class YoloDetector(
    context: Context,
    private val config: YoloConfig = YoloConfig(),
) : VisionModel {

    override val displayName = config.displayName

    private val interpreter: Interpreter
    private val inputWidth: Int
    private val inputHeight: Int
    private val inputIsFloat: Boolean

    private val inputBuffer: ByteBuffer
    private val inputFloatView: FloatBuffer?
    private val outputBuffer: ByteBuffer
    private val preprocessor: ImagePreprocessor
    private val inputFloats: FloatArray?
    private val inputBytes: ByteArray?

    // Output layout (decoded once from the output tensor shape).
    private val endToEnd: Boolean
    private val numAnchors: Int
    private val attrs: Int
    private val boxesFirst: Boolean
    private val hasObjectness: Boolean
    private val numClasses: Int

    // Output dtype handling.
    private val outIsFloat: Boolean
    private val outIsUnsigned: Boolean
    private val outScale: Float
    private val outZeroPoint: Int
    private val outElemSize: Int

    init {
        interpreter = Interpreter(
            loadMappedModel(context, config.modelAsset),
            Interpreter.Options().apply { numThreads = config.numThreads },
        )

        val inT = interpreter.getInputTensor(0)
        val inShape = inT.shape()
        require(inShape.size == 4 && inShape[0] == 1 && inShape[3] == 3) {
            "Expected YOLO input [1,H,W,3], got ${inShape.contentToString()}."
        }
        inputHeight = inShape[1]
        inputWidth = inShape[2]
        inputIsFloat = inT.dataType() == DataType.FLOAT32
        inputBuffer = ByteBuffer.allocateDirect(inT.numBytes()).order(ByteOrder.nativeOrder())
        preprocessor = ImagePreprocessor(inputWidth, inputHeight)
        if (inputIsFloat) {
            inputFloats = FloatArray(inputWidth * inputHeight * 3)
            inputFloatView = inputBuffer.asFloatBuffer()
            inputBytes = null
        } else {
            inputBytes = ByteArray(inputWidth * inputHeight * 3)
            inputFloats = null
            inputFloatView = null
        }

        val outT = interpreter.getOutputTensor(0)
        val outShape = outT.shape()
        require(outShape.size == 3 && outShape[0] == 1) {
            "Expected YOLO output [1,A,B], got ${outShape.contentToString()}."
        }
        numClasses = config.labels.size
        val gridNoObj = 4 + numClasses // v8
        val gridObj = 5 + numClasses // v5
        val d1 = outShape[1]
        val d2 = outShape[2]

        // Smaller axis is the per-prediction attribute count; larger is the anchor count.
        val attrAxis = minOf(d1, d2)
        val nAxis = maxOf(d1, d2)
        val matchesGrid = attrAxis == gridNoObj || attrAxis == gridObj
        val matchesE2e = attrAxis == E2E_ATTRS

        endToEnd = when (config.outputFormat) {
            YoloOutputFormat.GRID -> false
            YoloOutputFormat.END_TO_END -> true
            YoloOutputFormat.AUTO -> when {
                matchesE2e && !matchesGrid -> true // unambiguous (nc != 1 and nc != 2)
                matchesGrid && !matchesE2e -> false // unambiguous grid
                // Ambiguous: attr count 6 is both end-to-end and grid (nc=1 -> 5+nc,
                // nc=2 -> 4+nc). End-to-end heads emit few anchors (~300); grids emit
                // thousands -> use the anchor count to break the tie.
                matchesE2e && matchesGrid -> nAxis <= AUTO_E2E_MAX_ANCHORS
                else -> false // fall through to grid, which will throw if it truly doesn't fit
            }
        }

        if (endToEnd) {
            when {
                d2 == E2E_ATTRS -> { boxesFirst = false; attrs = d2; numAnchors = d1 }
                d1 == E2E_ATTRS -> { boxesFirst = true; attrs = d1; numAnchors = d2 }
                else -> throw IllegalArgumentException(
                    "END_TO_END expects an axis == $E2E_ATTRS " +
                        "(x1,y1,x2,y2,score,class_id), got ${outShape.contentToString()}.",
                )
            }
            hasObjectness = false
        } else {
            when {
                d1 == gridNoObj || d1 == gridObj -> { boxesFirst = true; attrs = d1; numAnchors = d2 }
                d2 == gridNoObj || d2 == gridObj -> { boxesFirst = false; attrs = d2; numAnchors = d1 }
                else -> throw IllegalArgumentException(
                    "GRID output ${outShape.contentToString()} doesn't fit $numClasses class(es): " +
                        "need an axis equal to $gridNoObj or $gridObj. Check YoloConfig.labels.",
                )
            }
            hasObjectness = attrs == gridObj
        }

        outIsFloat = outT.dataType() == DataType.FLOAT32
        outIsUnsigned = outT.dataType() == DataType.UINT8
        val qp = outT.quantizationParams()
        outScale = qp.scale
        outZeroPoint = qp.zeroPoint
        outElemSize = if (outIsFloat) 4 else 1
        outputBuffer = ByteBuffer.allocateDirect(outT.numBytes()).order(ByteOrder.nativeOrder())

        Log.i(
            TAG,
            "in=${inShape.contentToString()} ${inT.dataType()} " +
                "out=${outShape.contentToString()} ${outT.dataType()} " +
                "format=${if (endToEnd) "END_TO_END" else "GRID"} " +
                "anchors=$numAnchors attrs=$attrs objectness=$hasObjectness classes=$numClasses",
        )
    }

    override fun analyze(image: ImageProxy): VisionResult {
        val start = System.nanoTime()
        preprocessor.process(image)
        fillInput(preprocessor.pixels)

        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)

        val boxes = if (endToEnd) decodeEndToEnd() else decodeGrid()
        return VisionResult.Detection(
            boxes = boxes,
            frameWidth = preprocessor.uprightWidth,
            frameHeight = preprocessor.uprightHeight,
            inferenceTimeMs = (System.nanoTime() - start) / 1_000_000L,
        )
    }

    private fun fillInput(pixels: IntArray) {
        if (inputIsFloat) {
            val f = inputFloats!!
            var i = 0
            for (p in pixels) {
                f[i++] = ((p shr 16) and 0xFF) / 255f // R
                f[i++] = ((p shr 8) and 0xFF) / 255f  // G
                f[i++] = (p and 0xFF) / 255f          // B
            }
            val view = inputFloatView!!
            view.rewind()
            view.put(f) // writes through to inputBuffer; inputBuffer.position stays 0
        } else {
            val b = inputBytes!!
            var i = 0
            for (p in pixels) {
                b[i++] = ((p shr 16) and 0xFF).toByte()
                b[i++] = ((p shr 8) and 0xFF).toByte()
                b[i++] = (p and 0xFF).toByte()
            }
            inputBuffer.rewind()
            inputBuffer.put(b)
            inputBuffer.rewind()
        }
    }

    /** Reads attribute [attr] of anchor [anchor], dequantising quantised outputs. */
    private fun rawAt(attr: Int, anchor: Int): Float {
        val index = if (boxesFirst) attr * numAnchors + anchor else anchor * attrs + attr
        val offset = index * outElemSize
        return if (outIsFloat) {
            outputBuffer.getFloat(offset)
        } else {
            val q = if (outIsUnsigned) {
                outputBuffer.get(offset).toInt() and 0xFF
            } else {
                outputBuffer.get(offset).toInt() // int8
            }
            outScale * (q - outZeroPoint)
        }
    }

    // ---- End-to-end (YOLO26 / v10): [x1,y1,x2,y2,score,class_id], already NMS'd ----

    private fun decodeEndToEnd(): List<DetectionBox> {
        // Pass 1: gather rows above threshold (coords still in model units).
        val rows = ArrayList<FloatArray>()
        var maxCoord = 0f
        for (i in 0 until numAnchors) {
            val score = rawAt(4, i)
            if (score < config.confidenceThreshold) continue // also skips zero-padded rows
            val x1 = rawAt(0, i)
            val y1 = rawAt(1, i)
            val x2 = rawAt(2, i)
            val y2 = rawAt(3, i)
            maxCoord = maxOf(maxCoord, x1, y1, x2, y2)
            rows.add(floatArrayOf(x1, y1, x2, y2, score, rawAt(5, i)))
        }
        if (rows.isEmpty()) return emptyList()

        val pixelUnits = config.coordsArePixels ?: (maxCoord > 2f)
        val sx = if (pixelUnits) 1f / inputWidth else 1f
        val sy = if (pixelUnits) 1f / inputHeight else 1f

        val result = ArrayList<DetectionBox>(rows.size)
        for (r in rows) {
            val l = (r[0] * sx).coerceIn(0f, 1f)
            val t = (r[1] * sy).coerceIn(0f, 1f)
            val right = (r[2] * sx).coerceIn(0f, 1f)
            val bottom = (r[3] * sy).coerceIn(0f, 1f)
            if (right <= l || bottom <= t) continue
            val clsIdx = r[5].roundToInt()
            val label = config.labels.getOrElse(clsIdx) { "class $clsIdx" }
            result.add(DetectionBox(l, t, right, bottom, label, r[4]))
        }
        // Model already applied NMS; just sort and cap.
        result.sortByDescending { it.score }
        return if (result.size > config.maxDetections) {
            ArrayList(result.subList(0, config.maxDetections))
        } else {
            result
        }
    }

    // ---- Grid (v5/v8/v11): xywh-centre + per-class scores, needs NMS ----

    private class Cand(
        val cx: Float,
        val cy: Float,
        val w: Float,
        val h: Float,
        val score: Float,
        val cls: Int,
    )

    private fun decodeGrid(): List<DetectionBox> {
        val clsOffset = if (hasObjectness) 5 else 4
        val cands = ArrayList<Cand>()
        var maxCoord = 0f

        for (anchor in 0 until numAnchors) {
            val obj = if (hasObjectness) rawAt(4, anchor) else 1f
            // For standard YOLO exports class probs are <= 1, so obj is an upper
            // bound on the final score -> safe early reject.
            if (hasObjectness && obj < config.confidenceThreshold) continue

            var bestCls = 0
            var bestScore = 0f
            for (c in 0 until numClasses) {
                val s = rawAt(clsOffset + c, anchor)
                if (s > bestScore) {
                    bestScore = s
                    bestCls = c
                }
            }
            val score = obj * bestScore
            if (score < config.confidenceThreshold) continue

            val cx = rawAt(0, anchor)
            val cy = rawAt(1, anchor)
            val w = rawAt(2, anchor)
            val h = rawAt(3, anchor)
            maxCoord = maxOf(maxCoord, cx, cy, w, h)
            cands.add(Cand(cx, cy, w, h, score, bestCls))
        }
        if (cands.isEmpty()) return emptyList()

        // Auto-detect whether coords are normalised (0..1) or in input pixels.
        val pixelUnits = config.coordsArePixels ?: (maxCoord > 2f)
        val sx = if (pixelUnits) 1f / inputWidth else 1f
        val sy = if (pixelUnits) 1f / inputHeight else 1f

        val byClass = HashMap<Int, MutableList<DetectionBox>>()
        for (c in cands) {
            val ncx = c.cx * sx
            val ncy = c.cy * sy
            val nw = c.w * sx
            val nh = c.h * sy
            val l = (ncx - nw / 2f).coerceIn(0f, 1f)
            val t = (ncy - nh / 2f).coerceIn(0f, 1f)
            val r = (ncx + nw / 2f).coerceIn(0f, 1f)
            val b = (ncy + nh / 2f).coerceIn(0f, 1f)
            if (r <= l || b <= t) continue
            val label = config.labels.getOrElse(c.cls) { "class ${c.cls}" }
            byClass.getOrPut(c.cls) { ArrayList() }.add(DetectionBox(l, t, r, b, label, c.score))
        }

        val result = ArrayList<DetectionBox>()
        for (list in byClass.values) result.addAll(nms(list, config.iouThreshold))
        result.sortByDescending { it.score }
        return if (result.size > config.maxDetections) {
            ArrayList(result.subList(0, config.maxDetections))
        } else {
            result
        }
    }

    /** Greedy non-max suppression within a single class. */
    private fun nms(boxes: MutableList<DetectionBox>, iouTh: Float): List<DetectionBox> {
        boxes.sortByDescending { it.score }
        val kept = ArrayList<DetectionBox>()
        val removed = BooleanArray(boxes.size)
        for (i in boxes.indices) {
            if (removed[i]) continue
            val a = boxes[i]
            kept.add(a)
            for (j in i + 1 until boxes.size) {
                if (!removed[j] && iou(a, boxes[j]) > iouTh) removed[j] = true
            }
        }
        return kept
    }

    private fun iou(a: DetectionBox, b: DetectionBox): Float {
        val l = maxOf(a.left, b.left)
        val t = maxOf(a.top, b.top)
        val r = minOf(a.right, b.right)
        val bo = minOf(a.bottom, b.bottom)
        val inter = maxOf(0f, r - l) * maxOf(0f, bo - t)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    override fun close() {
        interpreter.close()
        preprocessor.close()
    }

    companion object {
        private const val TAG = "YoloDetector"

        /** End-to-end rows are [x1,y1,x2,y2,score,class_id]. */
        private const val E2E_ATTRS = 6

        /**
         * AUTO tie-break for the ambiguous attr==6 case (nc==1 v5-with-objectness, or
         * nc==2 v8): end-to-end heads emit ~300 anchors, grids emit thousands
         * (>=525 at imgsz 160, >=1029 at imgsz 224). 512 separates them for imgsz>=160.
         * Set YoloConfig.outputFormat explicitly to bypass the heuristic entirely.
         */
        private const val AUTO_E2E_MAX_ANCHORS = 512
    }
}

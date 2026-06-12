package com.example.tfliteclassifier

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View

/**
 * Transparent overlay that draws detection boxes on top of the camera preview.
 *
 * Boxes arrive normalised (0..1) against the upright frame; they are mapped with
 * the same FILL_CENTER transform [androidx.camera.view.PreviewView] uses by
 * default, so the boxes line up with what's shown (as long as preview and
 * analysis share an aspect ratio, which MainActivity arranges).
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var boxes: List<DetectionBox> = emptyList()
    private var frameWidth = 0
    private var frameHeight = 0

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.parseColor("#00E5FF")
    }
    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#CC0097A7")
    }
    private val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 38f
        typeface = Typeface.DEFAULT_BOLD
    }
    private val tmp = RectF()

    /** Replace the drawn boxes. Call on the main thread. */
    fun setDetections(boxes: List<DetectionBox>, frameWidth: Int, frameHeight: Int) {
        this.boxes = boxes
        this.frameWidth = frameWidth
        this.frameHeight = frameHeight
        invalidate()
    }

    /** Remove all boxes. Call on the main thread. */
    fun clear() {
        if (boxes.isEmpty()) return
        boxes = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (boxes.isEmpty() || frameWidth == 0 || frameHeight == 0) return

        // Replicate PreviewView's FILL_CENTER: scale to cover, then centre-crop.
        val scale = maxOf(width.toFloat() / frameWidth, height.toFloat() / frameHeight)
        val dx = (width - frameWidth * scale) / 2f
        val dy = (height - frameHeight * scale) / 2f

        for (b in boxes) {
            val l = b.left * frameWidth * scale + dx
            val t = b.top * frameHeight * scale + dy
            val r = b.right * frameWidth * scale + dx
            val bo = b.bottom * frameHeight * scale + dy
            tmp.set(l, t, r, bo)
            canvas.drawRect(tmp, boxPaint)

            val text = "${b.label} ${(b.score * 100).toInt()}%"
            val tw = labelTextPaint.measureText(text)
            val th = labelTextPaint.textSize
            val labelTop = (t - th - 14f).coerceAtLeast(0f)
            canvas.drawRect(l, labelTop, l + tw + 18f, labelTop + th + 12f, labelBgPaint)
            canvas.drawText(text, l + 9f, labelTop + th, labelTextPaint)
        }
    }
}

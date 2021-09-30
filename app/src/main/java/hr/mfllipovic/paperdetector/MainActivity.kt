package hr.mfllipovic.paperdetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.widget.SeekBar
import android.widget.SeekBar.OnSeekBarChangeListener
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import hr.mfllipovic.paperdetector.databinding.ActivityMainBinding
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc


class MainActivity : AppCompatActivity(), OpenCVCameraListener {

    companion object {
        private const val REQUEST_CODE_CAMERA_PERMISSION = 101
        private const val CANNY_LOWER_THRESHOLD: Double = 60.0
        private const val CANNY_UPPER_THRESHOLD: Double = 180.0
        private const val BLUR_KERNEL_SIZE: Double = 9.0
        private const val THRESHOLD_VALUE: Double = 190.0
        private val whiteColor: Scalar = Scalar(255.0, 255.0, 255.0)
        private val blueColor: Scalar = Scalar(60.0, 60.0, 255.0)
        private val green: Int = Color.parseColor("#006400")
    }


    private lateinit var mOpenCvCameraView: CameraBridgeViewBase
    private lateinit var binding: ActivityMainBinding

    private var drawContours = true
    private var drawConvexHull = true
    private var showProcessedImage = false
    private var epsilon = 10.0
    private var iterations = 3
    private var threshold = THRESHOLD_VALUE
    private var blurKernelSize = BLUR_KERNEL_SIZE
    private var kSize = 5.0
    private var cannyUpper = CANNY_UPPER_THRESHOLD
    private var cannyLower = CANNY_LOWER_THRESHOLD

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setupViews()
        setupOpenCV()
    }

    private fun setupOpenCV() {
        val version = OpenCVLoader.OPENCV_VERSION
        log("OpenCV version: $version")

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            when {
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED -> {
                    activateOpenCVCameraView()
                }
                shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {

                }
                else -> {
                    requestPermissions(
                        arrayOf(Manifest.permission.CAMERA), REQUEST_CODE_CAMERA_PERMISSION
                    )
                }
            }
        }
    }

    private fun setupViews() {
        with(binding) {
            epsilonValue.progress = epsilon.toInt()
            iterationsValue.progress = iterations
            thresholdValue.progress = threshold.toInt()
            ksizeValue.progress = kSize.toInt()
            cannyUpValue.progress = cannyUpper.toInt()
            cannyDownValue.progress = cannyLower.toInt()
            epsilonValue.setOnSeekBarChangeListener(object : SeekBarProgressListener() {
                override fun onChange(progress: Int) {
                    epsilon = progress.toDouble()
                }
            })
            iterationsValue.setOnSeekBarChangeListener(object : SeekBarProgressListener() {
                override fun onChange(progress: Int) {
                    iterations = progress
                }
            })
            thresholdValue.setOnSeekBarChangeListener(object : SeekBarProgressListener() {
                override fun onChange(progress: Int) {
                    threshold = progress.toDouble()
                }
            })
            ksizeValue.setOnSeekBarChangeListener(object : SeekBarProgressListener() {
                override fun onChange(progress: Int) {
                    kSize = progress.toDouble() + 1
                }
            })
            cannyUpValue.setOnSeekBarChangeListener(object : SeekBarProgressListener() {
                override fun onChange(progress: Int) {
                    cannyUpper = progress.toDouble()
                    if (progress <= cannyLower) {
                        cannyDownValue.progress = progress - 1
                    }
                }
            })
            cannyDownValue.setOnSeekBarChangeListener(object : SeekBarProgressListener() {
                override fun onChange(progress: Int) {
                    cannyLower = progress.toDouble()
                    if (progress >= cannyUpper) {
                        cannyUpValue.progress = progress + 1
                    }
                }
            })
            contours.setOnCheckedChangeListener { _, isChecked ->
                drawContours = isChecked
            }
            convexHull.setOnCheckedChangeListener { _, isChecked ->
                drawConvexHull = isChecked
            }
            previewType.setOnCheckedChangeListener { _, isChecked ->
                showProcessedImage = isChecked
            }

        }

    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            val indexOfCameraPermission = permissions.indexOf(Manifest.permission.CAMERA)
            if (indexOfCameraPermission != -1) {
                if (grantResults.isNotEmpty()) {
                    if (grantResults[indexOfCameraPermission] == PackageManager.PERMISSION_GRANTED) {
                        Toast.makeText(
                            applicationContext,
                            "Camera permission granted!",
                            Toast.LENGTH_LONG
                        ).show()
                        activateOpenCVCameraView()
                    } else {
                        Toast.makeText(
                            applicationContext,
                            "Camera permission is required to run this app!",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            }
        }
    }

    private fun activateOpenCVCameraView() {
        mOpenCvCameraView = binding.cameraView
        mOpenCvCameraView.setCameraPermissionGranted()
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK)
        mOpenCvCameraView.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView.setCvCameraViewListener(this)
        mOpenCvCameraView.enableView()
    }

    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    log("OpenCV loaded successfully")
                    activateOpenCVCameraView()
                }
                else -> super.onManagerConnected(status)
            }
        }
    }

    private fun log(message: String) {
        Log.i("MainActivity", message)
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        val originalImage = inputFrame!!.rgba()
        val convertedImage = inputFrame.gray()
        flipImages(convertedImage, originalImage)
        applyChangesToFrame(convertedImage)
        val bestContour = findBestContour(convertedImage)
        val finalImage = if (showProcessedImage) convertedImage else originalImage
        drawContours(finalImage, bestContour)
        drawOnScreen(finalImage, bestContour)
        return finalImage
    }

    private fun flipImages(convertedImage: Mat?, originalImage: Mat?) {
        Core.flip(convertedImage, convertedImage, -1)
        Core.flip(originalImage, originalImage, -1)
    }

    private fun findBestContour(convertedImage: Mat?): Contour {
        val contoursRaw = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            convertedImage, contoursRaw,
            Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
        )

        var bestContour = Contour()
        contoursRaw.forEach {
            Contour(it, epsilon).let { contour ->
                if (contour.betterThan(bestContour)) {
                    bestContour = contour
                }
            }
        }
        return bestContour
    }

    private fun drawContours(
        finalImage: Mat?,
        bestContour: Contour
    ) {
        if (bestContour.notEmpty()) {
            if (drawConvexHull) {
                Imgproc.drawContours(finalImage, mutableListOf(bestContour.hull), -1, blueColor, 12)
            }
            if (drawContours) {
                Imgproc.drawContours(
                    finalImage,
                    mutableListOf(bestContour.contour),
                    -1,
                    whiteColor,
                    5
                )
            }
        }
    }

    private fun drawOnScreen(
        finalImage: Mat,
        bestContour: Contour
    ) {
        val imgSize = finalImage.size().area()
        val isPaperPresent = (bestContour.area > (imgSize * 0.1)) && bestContour.corners == 4
        runOnUiThread {
            binding.paperDetectedValue.text = if (isPaperPresent) "Yes" else "No "
            binding.cornersCountValue.text = "${bestContour.corners}"
            binding.detectedLayout.setBackgroundColor(if (isPaperPresent) green else Color.BLACK)
            binding.cornersLayout.setBackgroundColor(if (isPaperPresent) green else Color.BLACK)
        }
    }

    private fun applyChangesToFrame(rgba: Mat?) {
        Imgproc.GaussianBlur(
            rgba,
            rgba,
            Size(blurKernelSize, blurKernelSize),
            0.0
        )
        Imgproc.threshold(
            rgba,
            rgba,
            threshold,
            255.0,
            Imgproc.THRESH_TRUNC
        )
        val kernel: Mat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kSize, kSize))
        val anchor = Point(-1.0, -1.0)
        Imgproc.dilate(rgba, rgba, kernel, anchor, iterations)
        Imgproc.erode(rgba, rgba, kernel, anchor, iterations)

        Imgproc.Canny(
            rgba,
            rgba,
            cannyLower,
            cannyUpper
        )
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            if (!OpenCVLoader.initDebug()) {
                log("Internal OpenCV library not found. Using OpenCV Manager for initialization")
                OpenCVLoader.initAsync(
                    OpenCVLoader.OPENCV_VERSION_3_0_0, this,
                    mLoaderCallback
                )
            } else {
                log("OpenCV library found inside package. Using it!")
                mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
            }
        }
    }

    override fun onDestroy() {
        mOpenCvCameraView.disableView()
        super.onDestroy()
    }
}

abstract class SeekBarProgressListener : OnSeekBarChangeListener {
    abstract fun onChange(progress: Int)

    override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) =
        onChange(progress)

    override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
    override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
}

interface OpenCVCameraListener : CameraBridgeViewBase.CvCameraViewListener2 {
    override fun onCameraViewStarted(width: Int, height: Int) = Unit
    override fun onCameraViewStopped() = Unit
}

class Contour() {
    var hull: MatOfPoint = MatOfPoint()
    var area: Double = 0.0
    var corners: Int = 0
    var contour: MatOfPoint = MatOfPoint()

    constructor(contour: MatOfPoint, epsilon: Double) : this() {
        if (!contour.empty()) {
            this.contour = contour
            val convexHull = MatOfInt()
            Imgproc.convexHull(contour, convexHull)
            val pointsOfHull = intsToPoints(
                contour,
                convexHull,
                ::MatOfPoint2f
            )
            val approxPoly = MatOfPoint2f()
            Imgproc.approxPolyDP(pointsOfHull, approxPoly, epsilon, true)
            area = Imgproc.contourArea(approxPoly)
            corners = approxPoly.toList().size
            hull = intsToPoints(contour, convexHull, ::MatOfPoint)
        }
    }

    fun betterThan(other: Contour): Boolean {
        return area > other.area
    }

    fun notEmpty(): Boolean {
        return !hull.empty() && !contour.empty()
    }

    private fun <T> intsToPoints(
        matOfPoint: MatOfPoint,
        matOfInt: MatOfInt,
        matFactory: (Array<Point?>) -> T
    ): T {
        val intsArray = matOfInt.toArray()
        val pointsArray = matOfPoint.toArray()
        val resultPointsArray = arrayOfNulls<Point>(intsArray.size)

        for (index in intsArray.indices) {
            resultPointsArray[index] = pointsArray[intsArray[index]]
        }
        return matFactory(resultPointsArray)
    }

}
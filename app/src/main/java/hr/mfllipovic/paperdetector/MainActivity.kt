package hr.mfllipovic.paperdetector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
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


class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    companion object {
        private const val REQUEST_CODE_CAMERA_PERMISSION = 101
        private const val CANNY_LOWER_THRESHOLD: Double = 60.0
        private const val CANNY_UPPER_THRESHOLD: Double = 180.0
        private const val BLUR_KERNEL_SIZE: Double = 3.0
        private const val THRESHOLD_VALUE: Double = 190.0
        private val whiteColor: Scalar = Scalar(255.0, 255.0, 255.0)
        private val greenColor: Scalar = Scalar(0.0, 255.0, 0.0)
    }


    private lateinit var mOpenCvCameraView: CameraBridgeViewBase
    private lateinit var mIntermediateMat: Mat
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val version = OpenCVLoader.OPENCV_VERSION
        log("version: $version")

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            when {
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED -> {
                    // You can use the API that requires the permission.
                    activateOpenCVCameraView()
                }
                shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                    // In an educational UI, explain to the user why your app requires this
                    // permission for a specific feature to behave as expected. In this UI,
                    // include a "cancel" or "no thanks" button that allows the user to
                    // continue using your app without granting the permission.
                }
                else -> {
                    // You can directly ask for the permission.
                    requestPermissions(
                        arrayOf(Manifest.permission.CAMERA),
                        REQUEST_CODE_CAMERA_PERMISSION
                    )
                }
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

    override fun onCameraViewStarted(width: Int, height: Int) {
        log("onCameraViewStarted")
        mIntermediateMat = Mat()
    }

    override fun onCameraViewStopped() {
        log("onCameraViewStopped")
        // Explicitly deallocate Mats
        mIntermediateMat.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        log("onCameraFrame")
        val rgba = inputFrame!!.rgba()

        Core.flip(rgba, rgba, -1)
//        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        Imgproc.cvtColor(rgba, mIntermediateMat, Imgproc.COLOR_BGR2GRAY)
//        frame = cv.GaussianBlur(frame, (self.noise_kernel_size, self.noise_kernel_size), 0)
        Imgproc.GaussianBlur(
            mIntermediateMat,
            mIntermediateMat,
            Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE),
            0.0
        )
        Imgproc.threshold(
            mIntermediateMat,
            mIntermediateMat,
            THRESHOLD_VALUE,
            255.0,
            Imgproc.THRESH_OTSU
        )

        val kernel: Mat = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        val anchor = Point(-1.0, -1.0)
        val iterations = 2
        Imgproc.dilate(mIntermediateMat, mIntermediateMat, kernel, anchor, iterations)
        Imgproc.erode(mIntermediateMat, mIntermediateMat, kernel, anchor, iterations)

        Imgproc.Canny(
            mIntermediateMat,
            mIntermediateMat,
            CANNY_LOWER_THRESHOLD,
            CANNY_UPPER_THRESHOLD
        )

        // find contours
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
            mIntermediateMat, contours,
            hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
        )
        var bestContour = MatOfPoint()
        var bestPaperArea = 0.0
        var bestPaperCorners = 0
        for (contour in contours) {
            val hull = MatOfInt()
            Imgproc.convexHull(contour, hull)
            val epsilon = 3.0
            val pointsOfHull = intsToPoints(contour, hull)
            val hullApprox = MatOfPoint2f()
            Imgproc.approxPolyDP(pointsOfHull, hullApprox, epsilon, true)

            val hullApproxArea = Imgproc.contourArea(hullApprox)
            if (hullApproxArea > bestPaperArea) {
                bestPaperArea = hullApproxArea
                bestPaperCorners = hullApprox.toList().size
                bestContour = contour
            }
        }

        val imgSize = rgba.size().area()
        val isPaperPresent = (bestPaperArea > (imgSize * 0.1)) && (bestPaperCorners == 4)
        Imgproc.putText(
            rgba,
            "Paper detected: ${if (isPaperPresent) "Yes" else "No"}",
            Point(10.0, 100.0),
            Imgproc.FONT_HERSHEY_PLAIN,
            7.0,
            if (isPaperPresent) greenColor else whiteColor,
            6
        )
        if (!bestContour.empty()) {
            Imgproc.drawContours(rgba, mutableListOf(bestContour), -1, whiteColor, 10)
        }
        return rgba
    }

    fun intsToPoints(matOfPoint: MatOfPoint, matOfInt: MatOfInt): MatOfPoint2f {
        val intsArray = matOfInt.toArray()
        val pointsArray = matOfPoint.toArray()
        val resultPointsArray = arrayOfNulls<Point>(intsArray.size)

        for (index in intsArray.indices) {
            resultPointsArray[index] = pointsArray[intsArray[index]]
        }
        val hull = MatOfPoint2f()
        hull.fromArray(*resultPointsArray)
        return hull
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


class Contour(
    var contour: MatOfPoint
) {
    var area: Double = Imgproc.contourArea(contour)
    var boundingRect: Rect = Imgproc.boundingRect(contour)

    fun percentOfBoxCovered(): Int {
        val boxWidth = this.boundingRect.width
        val boxHeight = this.boundingRect.height
        val boundingBoxArea = boxWidth * boxHeight
        val percentage = this.area / boundingBoxArea
        return (percentage * 100).toInt()
    }
}
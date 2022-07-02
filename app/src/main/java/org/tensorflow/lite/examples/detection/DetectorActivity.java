/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import static android.content.ContentValues.TAG;

import android.Manifest;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.camera2.CameraCharacteristics;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private int num=1;

  //load image

  private String filename;
  private int count=1;
  private final int SELECT_PICTURE=200;
  private Button fabloadimg;
//  private boolean LOAD_IMAGE=true;
  private int Load_previewWidth , Load_previewHeight;
  private int Load_targetW, Load_targetH;
  private int Load_cropW, Load_cropH;
  // here the preview image is drawn in portrait way
  private Bitmap Load_portraitBmp = null;
  // here the face is cropped and drawn
  private Bitmap Load_faceBmp = null;
  private Bitmap Load_rgbFrameBitmap = null;
  private Bitmap Load_croppedBitmap = null;
  private Bitmap Load_cropCopyBitmap = null;
  private Bitmap Load_bitmap = null;
  private Bitmap Load_finalBitmap = null;


  // FaceNet
//  private static final int TF_OD_API_INPUT_SIZE = 160;
//  private static final boolean TF_OD_API_IS_QUANTIZED = false;
//  private static final String TF_OD_API_MODEL_FILE = "facenet.tflite";
//  //private static final String TF_OD_API_MODEL_FILE = "facenet_hiroki.tflite";

  // MobileFaceNet
  private static final int TF_OD_API_INPUT_SIZE = 112;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";


  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  //private static final int CROP_SIZE = 320;
  //private static final Size CROP_SIZE = new Size(320, 320);


  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private SimilarityClassifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;
  private boolean addPending = false;
  //private boolean adding = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  //private Matrix cropToPortraitTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  // Face detector
  private FaceDetector faceDetector;

  // here the preview image is drawn in portrait way
  private Bitmap portraitBmp = null;
  // here the face is cropped and drawn
  private Bitmap faceBmp = null;

  private FloatingActionButton fabAdd;

  private HashMap<String, SimilarityClassifier.Recognition> knownFaces = new HashMap<>();


  private static final int CAMERA_PERMISSION_CODE = 100;
  private static final int STORAGE_PERMISSION_CODE = 101;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    checkPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE, STORAGE_PERMISSION_CODE);

//    if(LOAD_IMAGE){
//      fabloadimg= findViewById(R.id.fab_load);



//      fabloadimg.setOnClickListener(new View.OnClickListener() {
//        @Override
//        public void onClick(View view) {
//          imageChooser();
//        }
//      });


//    }

//    imageChooser();

    fabAdd = findViewById(R.id.fab_add);
    fabAdd.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        imageChooser();
//        onAddClick();
      }

    });


    // Real-time contour detection of multiple faces
    FaceDetectorOptions options =
            new FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                    .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                    .build();


    FaceDetector detector = FaceDetection.getClient(options);

    faceDetector = detector;



  }



  private void onAddClick() {

    addPending = true;
    //Toast.makeText(this, "click", Toast.LENGTH_LONG ).show();

  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);


    try {
      detector =
              TFLiteObjectDetectionAPIModel.create(
                      getAssets(),
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      //cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);


    int targetW, targetH;
    if (sensorOrientation == 90 || sensorOrientation == 270) {
      targetH = previewWidth;
      targetW = previewHeight;
    }
    else {
      targetW = previewWidth;
      targetH = previewHeight;
    }
    int cropW = (int) (targetW / 2.0);
    int cropH = (int) (targetH / 2.0);

    croppedBitmap = Bitmap.createBitmap(cropW, cropH, Config.ARGB_8888);

    portraitBmp = Bitmap.createBitmap(targetW, targetH, Config.ARGB_8888);
    faceBmp = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropW, cropH,
                    sensorOrientation, MAINTAIN_ASPECT);

//    frameToCropTransform =
//            ImageUtils.getTransformationMatrix(
//                    previewWidth, previewHeight,
//                    previewWidth, previewHeight,
//                    sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);


    Matrix frameToPortraitTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    targetW, targetH,
                    sensorOrientation, MAINTAIN_ASPECT);



    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                tracker.draw(canvas);
                if (isDebug()) {
                  tracker.drawDebug(canvas);
                }
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }


  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;

    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    InputImage image = InputImage.fromBitmap(croppedBitmap, 0);
    faceDetector
            .process(image)
            .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
              @Override
              public void onSuccess(List<Face> faces) {
                if (faces.size() == 0) {
                  updateResults(currTimestamp, new LinkedList<>());
                  return;
                }
                runInBackground(
                        new Runnable() {
                          @Override
                          public void run() {
                            onFacesDetected(currTimestamp, faces, addPending);
                            addPending = false;
                          }
                        });
              }

            });


  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }


  // Face Processing
  private Matrix createTransform(
          final int srcWidth,
          final int srcHeight,
          final int dstWidth,
          final int dstHeight,
          final int applyRotation) {

    Matrix matrix = new Matrix();
    if (applyRotation != 0) {
      if (applyRotation % 90 != 0) {
        LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
      }

      // Translate so center of image is at origin.
      matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

//        // Account for the already applied rotation, if any, and then determine how
//        // much scaling is needed for each axis.
//        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;
//        final int inWidth = transpose ? srcHeight : srcWidth;
//        final int inHeight = transpose ? srcWidth : srcHeight;

    if (applyRotation != 0) {

      // Translate back from origin centered reference to destination frame.
      matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
    }

    return matrix;

  }

  private void showAddFaceDialog(SimilarityClassifier.Recognition rec) {



        if (filename.isEmpty()) {
          return;
        }

        knownFaces.put(filename,rec);

    Toast.makeText(DetectorActivity.this, "Face added", Toast.LENGTH_SHORT).show();


    for(Map.Entry<String, SimilarityClassifier.Recognition> entry: knownFaces.entrySet()){
      detector.register(entry.getKey(),entry.getValue());
    }

  }

  private void updateResults(long currTimestamp, final List<SimilarityClassifier.Recognition> mappedRecognitions) {

    tracker.trackResults(mappedRecognitions, currTimestamp);
    trackingOverlay.postInvalidate();
    computingDetection = false;
    //adding = false;


    if (mappedRecognitions.size() > 0) {
      LOGGER.i("Adding results");
      SimilarityClassifier.Recognition rec = mappedRecognitions.get(0);
      if (rec.getExtra() != null) {
        showAddFaceDialog(rec);
      }

    }

    runOnUiThread(
            new Runnable() {
              @Override
              public void run() {
                showFrameInfo(previewWidth + "x" + previewHeight);
                showCropInfo(croppedBitmap.getWidth() + "x" + croppedBitmap.getHeight());
                showInference(lastProcessingTimeMs + "ms");
              }
            });

  }

  private void onFacesDetected(long currTimestamp, List<Face> faces, boolean add) {

    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
    final Canvas canvas = new Canvas(cropCopyBitmap);
    final Paint paint = new Paint();
    paint.setColor(Color.RED);
    paint.setStyle(Style.STROKE);
    paint.setStrokeWidth(2.0f);

    float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
    switch (MODE) {
      case TF_OD_API:
        minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
        break;
    }

    final List<SimilarityClassifier.Recognition> mappedRecognitions =
            new LinkedList<SimilarityClassifier.Recognition>();


    //final List<Classifier.Recognition> results = new ArrayList<>();

    // Note this can be done only once
    int sourceW = rgbFrameBitmap.getWidth();
    int sourceH = rgbFrameBitmap.getHeight();
    int targetW = portraitBmp.getWidth();
    int targetH = portraitBmp.getHeight();
    Matrix transform = createTransform(
            sourceW,
            sourceH,
            targetW,
            targetH,
            sensorOrientation);
    final Canvas cv = new Canvas(portraitBmp);

    // draws the original image in portrait mode.
    cv.drawBitmap(rgbFrameBitmap, transform, null);

    final Canvas cvFace = new Canvas(faceBmp);

    boolean saved = false;

    for (Face face : faces) {

//      LOGGER.i("FACE" + face.toString());
      LOGGER.i("Running detection on face " + currTimestamp);
      //results = detector.recognizeImage(croppedBitmap);

      final RectF boundingBox = new RectF(face.getBoundingBox());

      //final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
      final boolean goodConfidence = true; //face.get;
      if (boundingBox != null && goodConfidence) {

        // maps crop coordinates to original
        cropToFrameTransform.mapRect(boundingBox);

        // maps original coordinates to portrait coordinates
        RectF faceBB = new RectF(boundingBox);
        transform.mapRect(faceBB);

        // translates portrait to origin and scales to fit input inference size
        //cv.drawRect(faceBB, paint);
        float sx = ((float) TF_OD_API_INPUT_SIZE) / faceBB.width();
        float sy = ((float) TF_OD_API_INPUT_SIZE) / faceBB.height();
        Matrix matrix = new Matrix();
        matrix.postTranslate(-faceBB.left, -faceBB.top);
        matrix.postScale(sx, sy);

        cvFace.drawBitmap(portraitBmp, matrix, null);

        //canvas.drawRect(faceBB, paint);

        String label = "";
        float confidence = -1f;
        Integer color = Color.BLUE;
        Object extra = null;
        Bitmap crop = null;

//        LOGGER.i("Working");

        if (add) {
          crop = Bitmap.createBitmap(portraitBmp,
                  (int) faceBB.left,
                  (int) faceBB.top,
                  (int) faceBB.width(),
                  (int) faceBB.height());
        }


        final long startTime = SystemClock.uptimeMillis();
        final List<SimilarityClassifier.Recognition> resultsAux = detector.recognizeImage(faceBmp, add);
        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

        if (resultsAux.size() > 0) {

          SimilarityClassifier.Recognition result = resultsAux.get(0);

          extra = result.getExtra();
//          Object extra = result.getExtra();
//          if (extra != null) {
//            LOGGER.i("embeeding retrieved " + extra.toString());
//          }

          float conf = result.getDistance();
          if (conf < 1.0f) {

            confidence = conf;
            label = result.getTitle();
            if (result.getId().equals("0")) {
              color = Color.GREEN;
            }
            else {
              color = Color.RED;
            }
          }

        }

        if (getCameraFacing() == CameraCharacteristics.LENS_FACING_FRONT) {

          // camera is frontal so the image is flipped horizontally
          // flips horizontally
          Matrix flip = new Matrix();
          if (sensorOrientation == 90 || sensorOrientation == 270) {
            flip.postScale(1, -1, previewWidth / 2.0f, previewHeight / 2.0f);
          }
          else {
            flip.postScale(-1, 1, previewWidth / 2.0f, previewHeight / 2.0f);
          }
          //flip.postScale(1, -1, targetW / 2.0f, targetH / 2.0f);
          flip.mapRect(boundingBox);

        }

        final SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                "0", label, confidence, boundingBox);

        result.setColor(color);
        result.setLocation(boundingBox);
        result.setExtra(extra);
        result.setCrop(crop);
        mappedRecognitions.add(result);

      }


    }

    //    if (saved) {
//      lastSaved = System.currentTimeMillis();
//    }

    updateResults(currTimestamp, mappedRecognitions);


  }

  public void savetoBitmap(final Bitmap image, final String filename) {
//    final String root =
//        Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "tensorflow";
//    LOGGER.i("Saving %dx%d bitmap to %s.", bitmap.getWidth(), bitmap.getHeight(), root);
//    final File myDir = new File(root);
//
////    ContextWrapper cw = new ContextWrapper(cw.getApplicationContext());
////    // path to /data/data/yourapp/app_data/imageDir
////    File myDir = cw.getDir("imageDir", Context.MODE_PRIVATE);
//
//    if (!myDir.mkdirs()) {
//      LOGGER.i("Make dir failed");
//    }
//
//    final String fname = filename;
//    final File file = new File(myDir, fname);
//    if (file.exists()) {
//      file.delete();
//    }
//    try {
//      final FileOutputStream out = new FileOutputStream(file);
//      bitmap.compress(Bitmap.CompressFormat.PNG, 99, out);
//      out.flush();
//      out.close();
//    } catch (final Exception e) {
//      LOGGER.e(e, "Exception!");
//    }

    File pictureFile = getOutputMediaFile();
    if (pictureFile == null) {
      Log.d(TAG,
              "Error creating media file, check storage permissions: ");
//              e.getMessage());
      return;
    }
    try {
      FileOutputStream fos = new FileOutputStream(pictureFile);
      image.compress(Bitmap.CompressFormat.PNG, 90, fos);
      Log.d(TAG,
              "image saved ");
      fos.close();
    } catch (FileNotFoundException e) {
      Log.d(TAG, "File not found: " + e.getMessage());
    } catch (IOException e) {
      Log.d(TAG, "Error accessing file: " + e.getMessage());
    }
  }

  /** Create a File for saving an image or video */
  private File getOutputMediaFile(){
    // To be safe, you should check that the SDCard is mounted
    // using Environment.getExternalStorageState() before doing this.
    File mediaStorageDir = new File(Environment.getExternalStorageDirectory()
            + "/Android/data/"
            + getApplicationContext().getPackageName()
            + "/tensorflowFiles");

    // This location works best if you want the created images to be shared
    // between applications and persist after your app has been uninstalled.

    // Create the storage directory if it does not exist
    if (! mediaStorageDir.exists()){
      if (! mediaStorageDir.mkdirs()){
        return null;
      }
    }

    // Create a media file name
    String timeStamp = new SimpleDateFormat("ddMMyyyy_HHmm").format(new Date());
    File mediaFile;
    String mImageName="MI_"+ timeStamp +".jpg";
    mediaFile = new File(mediaStorageDir.getPath() + File.separator + mImageName);
    return mediaFile;
  }


  public void checkPermission(String permission, int requestCode)
  {
    if (ContextCompat.checkSelfPermission(DetectorActivity.this, permission) == PackageManager.PERMISSION_DENIED) {

      // Requesting the permission
      ActivityCompat.requestPermissions(DetectorActivity.this, new String[] { permission }, requestCode);
    }
    else {
      Toast.makeText(DetectorActivity.this, "Permission already granted", Toast.LENGTH_SHORT).show();
    }
  }

  // This function is called when the user accepts or decline the permission.
  // Request Code is used to check which permission called this function.
  // This request code is provided when the user is prompt for permission.

  @Override
  public void onRequestPermissionsResult(int requestCode,
                                         @NonNull String[] permissions,
                                         @NonNull int[] grantResults)
  {
    super.onRequestPermissionsResult(requestCode,
            permissions,
            grantResults);

    if (requestCode == CAMERA_PERMISSION_CODE) {
      if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        Toast.makeText(DetectorActivity.this, "Camera Permission Granted", Toast.LENGTH_SHORT) .show();
      }
      else {
        Toast.makeText(DetectorActivity.this, "Camera Permission Denied", Toast.LENGTH_SHORT) .show();
      }
    }
    else if (requestCode == STORAGE_PERMISSION_CODE) {
      if (grantResults.length > 0
              && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        Toast.makeText(DetectorActivity.this, "Storage Permission Granted", Toast.LENGTH_SHORT).show();
      } else {
        Toast.makeText(DetectorActivity.this, "Storage Permission Denied", Toast.LENGTH_SHORT).show();
      }
    }
  }


  void imageChooser() {

    // create an instance of the
    // intent of the type image
    Intent i = new Intent();
    i.setType("image/*");
    i.setAction(Intent.ACTION_GET_CONTENT);

    // pass the constant to compare it
    // with the returned requestCode

    startActivityForResult(Intent.createChooser(i, "Select Picture"), SELECT_PICTURE);
  }

  // this function is triggered when user
  // selects the image from the imageChooser
  public void onActivityResult(int requestCode, int resultCode, Intent data) {
    super.onActivityResult(requestCode, resultCode, data);

    final long currTimestamp = timestamp;

    if (resultCode == RESULT_OK) {

      // compare the resultCode with the
      // SELECT_PICTURE constant
      if (requestCode == SELECT_PICTURE) {
        // Get the url of the image from data
        Uri selectedImageUri = data.getData();
        if (null != selectedImageUri) {
          // update the preview image in the layout
          filename=getFileName(selectedImageUri);
          try {
            Load_bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
//            ImageUtils.saveBitmap(bitmap,"Curry" );
//            Toast.makeText(DetectorActivity.this, "image from storage is taken as an input", Toast.LENGTH_SHORT).show();
            LOGGER.i("Image from storage is taken as an input");


            Load_targetH=Load_previewHeight= Load_bitmap.getHeight();
            Load_targetW=Load_previewWidth= Load_bitmap.getWidth();

            Load_cropH=(int)(Load_targetH/2.0);
            Load_cropW=(int)(Load_targetW/2.0);

            LOGGER.i("width:" + Load_previewWidth+" height: "+Load_previewHeight);


            Load_rgbFrameBitmap = Bitmap.createBitmap(Load_previewWidth, Load_previewHeight, Config.ARGB_8888);

//            Load_rgbFrameBitmap.setPixels(getRgbBytes(), 0, Load_previewWidth, 0, 0, Load_previewWidth, Load_previewHeight);
            LOGGER.i("after Loaded rgbFrameBitmap");
//            final Canvas canvas = new Canvas(Load_bitmap);
//            canvas.drawBitmap(Load_rgbFrameBitmap, frameToCropTransform, null);

            String ABC="ARGB_8888";




            Load_croppedBitmap = Bitmap.createBitmap(Load_cropW, Load_cropH, Config.ARGB_8888);
            Load_portraitBmp = Bitmap.createBitmap(Load_targetW, Load_targetH, Config.ARGB_8888);
            Load_faceBmp = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);


//            ImageUtils.saveBitmap(Load_bitmap,"100000" );
//            ImageUtils.saveBitmap(Load_croppedBitmap,"20000" );
//            ImageUtils.saveBitmap(Load_portraitBmp,"30000" );
//            ImageUtils.saveBitmap(Load_faceBmp,"40000" );

            LOGGER.i("before input image");

            Load_finalBitmap = Load_bitmap.copy(Bitmap.Config.ARGB_8888, false);
            InputImage image = InputImage.fromBitmap(Load_finalBitmap, 0);

            LOGGER.i("after input image");

            faceDetector
                    .process(image)
                    .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                      @Override
                      public void onSuccess(List<Face> faces) {
                        LOGGER.i("after Process(image)");

                        if (faces.size() == 0) {
//                          Toast.makeText(DetectorActivity.this, "no face found", Toast.LENGTH_SHORT).show();
                          LOGGER.i("no face found");
                        }
                        else {
                          LOGGER.i("face found "+ faces.size());

                          onfaceDetectedFromLoadedImage(currTimestamp,faces);
                        }
                      }

                    });

          } catch (IOException e) {
            e.printStackTrace();
          }
        }
      }
    }
  }



  private void onfaceDetectedFromLoadedImage( final long currTimestamp , List<Face> faces) {


    Load_cropCopyBitmap = Bitmap.createBitmap(Load_croppedBitmap);

//    ImageUtils.saveBitmap(Load_cropCopyBitmap,"50000" );

//    final Canvas canvas = new Canvas(Load_cropCopyBitmap);
//    final Paint paint = new Paint();
//    paint.setColor(Color.RED);
//    paint.setStyle(Style.STROKE);
//    paint.setStrokeWidth(2.0f);
//
//    float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
//    switch (MODE) {
//      case TF_OD_API:
//        minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
//        break;
//    }

    final List<SimilarityClassifier.Recognition> mappedRecognitions =
            new LinkedList<SimilarityClassifier.Recognition>();


    final List<SimilarityClassifier.Recognition> results = new ArrayList<>();

    // Note this can be done only once
//    int sourceW = Load_rgbFrameBitmap.getWidth();
//    int sourceH = Load_rgbFrameBitmap.getHeight();
//    int targetW = Load_portraitBmp.getWidth();
//    int targetH = Load_portraitBmp.getHeight();
//    Matrix transform = createTransform(
//            sourceW,
//            sourceH,
//            targetW,
//            targetH,
//            sensorOrientation);
//    final Canvas cv = new Canvas(Load_portraitBmp);
//
//    // draws the original image in portrait mode.
//    cv.drawBitmap(Load_rgbFrameBitmap, transform, null);
//
//    final Canvas cvFace = new Canvas(Load_faceBmp);
//
//    boolean saved = false;

    {
      Face face=faces.get(0);
      LOGGER.i("FACE" + face.toString());
      LOGGER.i("Running detection on face ");
      //results = detector.recognizeImage(croppedBitmap);

      final RectF boundingBox = new RectF(face.getBoundingBox());
      LOGGER.i("Bounding Box : " + boundingBox);

      //final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
      final boolean goodConfidence = true; //face.get;
      if (boundingBox != null && goodConfidence) {

        LOGGER.i("Testing");
        // maps crop coordinates to original
//        cropToFrameTransform.mapRect(boundingBox);

        // maps original coordinates to portrait coordinates
        RectF faceBB = new RectF(boundingBox);
//        transform.mapRect(faceBB);
        LOGGER.i("Tesdting: "+ faceBB);

        // translates portrait to origin and scales to fit input inference size
        //cv.drawRect(faceBB, paint);

        LOGGER.i("Testing: width " + faceBB.width() + " " +faceBB.height());

//        float sx = ((float) TF_OD_API_INPUT_SIZE) / faceBB.width();
//        float sy = ((float) TF_OD_API_INPUT_SIZE) / faceBB.height();
//        Matrix matrix = new Matrix();
//        matrix.postTranslate(-faceBB.left, -faceBB.top);
//        matrix.postScale(sx, sy);

//        cvFace.drawBitmap(Load_portraitBmp, matrix, null);

        LOGGER.i("cvFace");

        //canvas.drawRect(faceBB, paint);

        String label = "";
        float confidence = -1f;
        Integer color = Color.BLUE;
        Object extra = null;
        Bitmap crop = null;

        boolean add= true;
        if (add) {
          crop = Bitmap.createBitmap(Load_bitmap,
                  (int) faceBB.left,
                  (int) faceBB.top,
                  (int) faceBB.width(),
                  (int) faceBB.height());
        }

        LOGGER.i("Testing after creating crop");


        LOGGER.i("Testing");

//        final long startTime = SystemClock.uptimeMillis();
        final List<SimilarityClassifier.Recognition> resultsAux = detector.recognizeImage(crop, add);
//        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

        if (resultsAux.size() > 0) {

          SimilarityClassifier.Recognition result = resultsAux.get(0);

          extra = result.getExtra();
//          Object extra = result.getExtra();
//          if (extra != null) {
//            LOGGER.i("embeeding retrieved " + extra.toString());
//          }

          float conf = result.getDistance();
          if (conf < 1.0f) {

            confidence = conf;
            label = result.getTitle();
            if (result.getId().equals("0")) {
              color = Color.GREEN;
            }
            else {
              color = Color.RED;
            }
          }

        }

//        if (getCameraFacing() == CameraCharacteristics.LENS_FACING_FRONT) {
//
//          // camera is frontal so the image is flipped horizontally
//          // flips horizontally
//          Matrix flip = new Matrix();
//          if (sensorOrientation == 90 || sensorOrientation == 270) {
//            flip.postScale(1, -1, Load_previewWidth / 2.0f, Load_previewHeight / 2.0f);
//          }
//          else {
//            flip.postScale(-1, 1, Load_previewWidth / 2.0f, Load_previewHeight / 2.0f);
//          }
//          //flip.postScale(1, -1, targetW / 2.0f, targetH / 2.0f);
//          flip.mapRect(boundingBox);
//
//        }

        final SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                "0", label, confidence, boundingBox);

        result.setColor(color);
        result.setLocation(boundingBox);
        result.setExtra(extra);
        result.setCrop(crop);
        mappedRecognitions.add(result);

      }

      if (mappedRecognitions.size() > 0) {
        LOGGER.i("Adding results");
        SimilarityClassifier.Recognition rec = mappedRecognitions.get(0);
        if (rec.getExtra() != null) {
          showAddFaceDialog(rec);
        }

      }

    }
    Toast.makeText(DetectorActivity.this, "out of on face detected from loaded image " + knownFaces.size() , Toast.LENGTH_SHORT).show();

  }


  public String getFileName(Uri uri) {
    String result = null;
    if (uri.getScheme().equals("content")) {
      Cursor cursor = getContentResolver().query(uri, null, null, null, null);
      try {
        if (cursor != null && cursor.moveToFirst()) {
          result = cursor.getString(cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME));
        }
      } finally {
        cursor.close();
      }
    }
    if (result == null) {
      result = uri.getPath();
      int cut = result.lastIndexOf('/');
      if (cut != -1) {
        result = result.substring(cut + 1);
      }
    }
    return result;
  }

}

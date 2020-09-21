/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tracking;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Pair;
import android.util.TypedValue;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import org.tensorflow.lite.examples.detection.BodyPart;
import org.tensorflow.lite.examples.detection.KeyPoint;
import org.tensorflow.lite.examples.detection.OffsetPosition;
import org.tensorflow.lite.examples.detection.Person;
import org.tensorflow.lite.examples.detection.Position;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker {
  private static final float TEXT_SIZE_DIP = 18;
  private static final float MIN_SIZE = 16.0f;
  private static final int[] COLORS = {
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.CYAN,
    Color.MAGENTA,
    Color.WHITE,
    Color.parseColor("#55FF55"),
    Color.parseColor("#FFA500"),
    Color.parseColor("#FF8888"),
    Color.parseColor("#AAAAFF"),
    Color.parseColor("#FFFFAA"),
    Color.parseColor("#55AAAA"),
    Color.parseColor("#AA33AA"),
    Color.parseColor("#0D0068")
  };
  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
  private final Paint boxPaint = new Paint();
  private final float textSizePx;
  private final BorderedText borderedText;
  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;


  final List<Pair<BodyPart, BodyPart>> bodyJoints = new LinkedList<Pair<BodyPart, BodyPart>>();

  Pair pair1 = new Pair(BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW);
  Pair pair2 = new Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER);
  Pair pair3 = new Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER);
  Pair pair4 = new Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW);
  Pair pair5 = new Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST);
  Pair pair6 = new Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP);
  Pair pair7 = new Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP);
  Pair pair8 = new Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER);
  Pair pair9 = new Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE);
  Pair pair10 = new Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE);
  Pair pair11 = new Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE);
  Pair pair12 = new Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE);

  public void getBodyJoints(){
    bodyJoints.add(pair1);
    bodyJoints.add(pair2);
    bodyJoints.add(pair3);
    bodyJoints.add(pair4);
    bodyJoints.add(pair5);
    bodyJoints.add(pair6);
    bodyJoints.add(pair7);
    bodyJoints.add(pair8);
    bodyJoints.add(pair9);
    bodyJoints.add(pair10);
    bodyJoints.add(pair11);
    bodyJoints.add(pair12);
  }

  // keypoint得分阈值
  private float minScore = 0.5f;
  // 关键点的半径大小
  private float circleRadius = 5.0f;


  public MultiBoxTracker(final Context context) {
    for (final int color : COLORS) {
      availableColors.add(color);
    }

    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(10.0f);
    boxPaint.setStrokeCap(Cap.ROUND);
    boxPaint.setStrokeJoin(Join.ROUND);
    boxPaint.setStrokeMiter(100);

    textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  public synchronized void setFrameConfiguration(
      final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = detection.second;
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, List<Person> persons, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results, persons);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void draw(final Canvas canvas) {
    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
        Math.min(
            canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
            canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
        ImageUtils.getTransformationMatrix(
            frameWidth,
            frameHeight,
            (int) (multiplier * (rotated ? frameHeight : frameWidth)),
            (int) (multiplier * (rotated ? frameWidth : frameHeight)),
            sensorOrientation,
            false);

    // 设置坐标换算比例
    float scaleRatioX = canvas.getWidth() / 300.0f;
    float scaleRatioY = (canvas.getWidth() * frameWidth/ frameHeight) / 300.0f;

    for (final TrackedRecognition recognition : trackedObjects) {
      final RectF trackedPos = new RectF(recognition.location);

      getFrameToCanvasMatrix().mapRect(trackedPos);
      boxPaint.setColor(recognition.color);

      float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
      canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

      // 设置对抠出的人像进行坐标换算的比例。换算后的坐标为300*300范围内的坐标值。
      float sizeRatio = (recognition.scaleSize / 257.0f);

//      画出特征点
      List<KeyPoint> keyPoints = recognition.keyPoints;
      for (final KeyPoint keypoint: keyPoints){
        if(keypoint.getScore() > minScore){
          Position position = keypoint.getPosition();
          // 调整特征点的坐标，坐标值为300*300内的坐标。
          float adjustedX = ((position.getX()*sizeRatio) + recognition.offset.getX());
          float adjustedY = ((position.getY()*sizeRatio) + recognition.offset.getY());
          canvas.drawCircle(
                  adjustedX * scaleRatioX,
                  adjustedY * scaleRatioY,
                  circleRadius,
                  boxPaint
          );
        }
      }

      // 身体连线
//      getBodyJoints();
//      float offsetX = recognition.offset.getX();
//      float offsetY = recognition.offset.getY();
//      for (final Pair<BodyPart, BodyPart> pair : bodyJoints){
//        if((recognition.keyPoints.get(pair.first.ordinal()).getScore() > minScore)
//                && (recognition.keyPoints.get(pair.second.ordinal()).getScore() > minScore)
//        ){
//          float firstX = (recognition.keyPoints.get(pair.first.ordinal()).getPosition().getX());
//          float firstY = (recognition.keyPoints.get(pair.first.ordinal()).getPosition().getY());
//          float secondX = (recognition.keyPoints.get(pair.second.ordinal()).getPosition().getX());
//          float secondY = (recognition.keyPoints.get(pair.second.ordinal()).getPosition().getY());
//          canvas.drawLine(
//                  (firstX*sizeRatio + offsetX) * scaleRatioX,
//                  (firstY*sizeRatio + offsetY) * scaleRatioY,
//                  (secondX*sizeRatio + offsetX) * scaleRatioX,
//                  (secondY*sizeRatio + offsetY) * scaleRatioY,
//                  boxPaint
//
//          );
//        }
//      }

      final String labelString =
          !TextUtils.isEmpty(recognition.title)
              ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence))
              : String.format("%.2f", (100 * recognition.detectionConfidence));
      //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
      // labelString);
      borderedText.drawText(
          canvas, trackedPos.left + cornerSize, trackedPos.top, labelString + "%", boxPaint);
    }
  }

  private void processResults(final List<Recognition> results, final List<Person> persons) {
    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();
    final List<Pair<Recognition, Person>> personToTrack = new LinkedList<Pair<Recognition, Person>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    int i = 0;
    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
          "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
      personToTrack.add(new Pair<Recognition, Person>(result, persons.get(i)));

      i++;
    }

    trackedObjects.clear();
    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    int j = 0;
    for (final Pair<Float, Recognition> potential : rectsToTrack) {
      final TrackedRecognition trackedRecognition = new TrackedRecognition();
      trackedRecognition.detectionConfidence = potential.first;
      trackedRecognition.location = new RectF(potential.second.getLocation());
      trackedRecognition.title = potential.second.getTitle();
      trackedRecognition.color = COLORS[trackedObjects.size()];
      trackedRecognition.keyPoints = persons.get(j).getKeyPoints();
      trackedRecognition.offset = persons.get(j).getOffset();
      trackedRecognition.scaleSize = persons.get(j).getScaleSize();
      trackedObjects.add(trackedRecognition);

      if (trackedObjects.size() >= COLORS.length) {
        break;
      }
      j++;
    }
  }

  private static class TrackedRecognition {
    RectF location;
    float detectionConfidence;
    int color;
    String title;
    List<KeyPoint> keyPoints;
    float scaleSize;
    OffsetPosition offset;
  }
}

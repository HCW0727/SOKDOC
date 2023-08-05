from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle



def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image



# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='/Users/huhchaewon/Python_Projects/HNW/model/pose_estimation.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.


cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()

    if ret == False:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(rgb_image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(rgb_image.numpy_view(), detection_result)

    if detection_result.pose_landmarks:
      
      # Get landmarks
      landmarks = detection_result.pose_landmarks[0]

      # Get coordinates
      shoulder_left = [landmarks[11].x, landmarks[11].y]
      elbow_left = [landmarks[13].x, landmarks[13].y]
      wrist_left = [landmarks[15].x, landmarks[15].y]

      shoulder_right = [landmarks[12].x, landmarks[12].y]
      elbow_right = [landmarks[14].x, landmarks[14].y]
      wrist_right = [landmarks[16].x, landmarks[16].y]

      # Calculate the angle
      angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
      angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)

      # # Print the angle
      # print("Angle left: ", angle_left)
      # print("Angle right: ", angle_right)
      difference_value = abs(angle_left - angle_right)
      print("Angle difference: ", abs(angle_left - angle_right))

      if difference_value > 20:
        cv2.putText(annotated_image, str(difference_value) , (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

      else:
        cv2.putText(annotated_image, str(difference_value) , (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 5)

    cv2.imshow('result',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(10)

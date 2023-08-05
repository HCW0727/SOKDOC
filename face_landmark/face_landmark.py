from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


landmarker_path = '/Users/huhchaewon/Python_Projects/HNW/model/face_landmark.task'
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path=landmarker_path)

options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


# ... (위의 코드와 동일)
# STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.jpg")


cap = cv2.VideoCapture(0)  # 웹캠에서 이미지를 가져옵니다.
while True:
    
  ret, frame = cap.read()  # 웹캠에서 프레임을 읽어옵니다.

  if ret == False:
    break

  frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

  # STEP 4: Detect face landmarks from the input image.
  detection_result = detector.detect(rgb_image)

  # STEP 5: Process the detection result. In this case, visualize it.

  print('result : ',detection_result)
  # print(detection_result.shape)
  annotated_image = draw_landmarks_on_image(rgb_image.numpy_view(), detection_result)
  cv2.imshow('result',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
  
  if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료합니다.
        break


#############

import cv2

# OpenCV의 imread를 이용하여 이미지를 읽습니다.
# image_path는 여러분의 이미지 파일 경로로 변경해야 합니다.
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

for detection in detection_result.detections:
    bounding_box = detection.bounding_box
    # 왼쪽 위 픽셀 좌표
    left_top = (bounding_box.origin_x, bounding_box.origin_y)
    # 오른쪽 아래 픽셀 좌표
    right_bottom = (bounding_box.origin_x + bounding_box.width, bounding_box.origin_y + bounding_box.height)
    
    # 이미지를 자르기 위해 좌표를 사용합니다.
    cropped_image = image[int(left_top[1]):int(right_bottom[1]), int(left_top[0]):int(right_bottom[0])]

    # 자른 이미지를 확인하려면 OpenCV의 imshow를 사용합니다.
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)

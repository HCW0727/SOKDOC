from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from PIL import Image

central_landmarks = [10, 152, 8, 165, 159, 144, 133, 155, 154, 153, 145, 7, 338, 151, 168, 6, 197, 195, 5, 4, 321, 61, 291, 306, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 179, 316, 315, 17, 84, 181, 91, 146, 92, 93, 180, 85, 295, 279, 269, 267, 0, 296, 332, 282, 20, 19, 37, 40, 39, 289, 286, 332, 297, 332, 284]

# def draw_landmarks_on_image(rgb_image, detection_result):
#   face_landmarks_list = detection_result.face_landmarks
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected faces to visualize.
#   for idx in range(len(face_landmarks_list)):
#     face_landmarks = face_landmarks_list[idx]

#     # Draw the face landmarks.
#     face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     face_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
#     ])

#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_tesselation_style())
#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_contours_style())
#     solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_IRISES,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp.solutions.drawing_styles
#           .get_default_face_mesh_iris_connections_style())

#   return annotated_image

def draw_landmarks_on_image(rgb_image, detection_result):
  image_height,image_width, _ = rgb_image.shape
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

    # draw red dots on landmarks 4, 159, 145
    for landmark_id in [4, 159, 145]:
      landmark = face_landmarks_proto.landmark[landmark_id]
      x = int(landmark.x * image_width)
      y = int(landmark.y * image_height)
      cv2.circle(annotated_image, (x, y), 5, (255, 100, 255), -1)

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
base_options = python.BaseOptions(model_asset_path='/Users/huhchaewon/Python_Projects/HNW/model/face_landmark.task')

options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)



def run_face_landmark(image):
  # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  # image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
  # image = mp.Image.create_from_file("dog.jpeg")

  # STEP 4: Detect face landmarks from the input image.
  detection_result = detector.detect(image)

  return detection_result


def asy_val(detection_result_FL,rotated_image):
  height,width,_ = rotated_image.shape 
  for face_landmarks in detection_result_FL.face_landmarks:
    center_landmark = (face_landmarks[13].x * width,(face_landmarks[13].y + face_landmarks[15].y) / 2 * height)
    right_landmark = (face_landmarks[375].x * width,face_landmarks[375].y * height)
    left_landmark = (face_landmarks[146].x * width,face_landmarks[146].y * height)

    modified_right_landmark = (right_landmark[0] - center_landmark[0], right_landmark[1] - center_landmark[1])

    # print(modified_right_landmark)

    # print('angle : ',angle_between_points(left_landmark, center_landmark,modified_right_landmark))
    value = 100 - angle_between_points(left_landmark, center_landmark,modified_right_landmark)
    min_value, max_value = 15, 27
    normalized_value = (value - min_value) / (max_value - min_value)
    mapped_value = normalized_value * 100


    # return mapped_value
    return value

def angle_between_points(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)

    
    cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    angle = np.arccos(cos_angle)

    return np.degrees(angle)

# def rotate_img(face_landmarks, cropped_image):
#   image_height, image_width,_ = cropped_image.shape
#    # 코의 랜드마크 인덱스는 1, 입의 랜드마크 인덱스는 17입니다.
   

#   print(face_landmarks)
#   nose_landmark = face_landmarks[1]
#   mouth_landmark = face_landmarks[17]

#   # 픽셀 좌표로 변환합니다.
#   nose_x = int(nose_landmark.x * image_width)
#   nose_y = int(nose_landmark.y * image_height)
#   mouth_x = int(mouth_landmark.x * image_width)
#   mouth_y = int(mouth_landmark.y * image_height)

#   # 랜드마크 간의 각도를 계산합니다.
#   dy = mouth_y - nose_y
#   dx = mouth_x - nose_x
#   angle = np.arctan2(dy, dx) * 180. / np.pi

#   # 회전 행렬을 계산합니다.
#   rotation_matrix = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), -angle, 1)

#   # 이미지를 회전시킵니다.
#   rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (image_width, image_height))

#   cv2.imshow('test',rotated_image)
#   cv2.waitKey(0)

def trignometry_for_distance(a, b):
    return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) + ((b[1] - a[1]) * (b[1] - a[1])))


def align_face(detection_result_FL, image):
    image_height, image_width, _ = image.shape
    
    for face_landmarks in detection_result_FL.face_landmarks:
        left_eye_landmark = face_landmarks[159]
        right_eye_landmark = face_landmarks[386]

        # 눈의 중심을 계산합니다.
        eyes_center_x = (left_eye_landmark.x + right_eye_landmark.x) / 2
        eyes_center_y = (left_eye_landmark.y + right_eye_landmark.y) / 2

        # 눈의 중심 위치를 픽셀 좌표로 변환합니다.
        eyes_center_x_pixel = int(eyes_center_x * image_width)
        eyes_center_y_pixel = int(eyes_center_y * image_height)

        # 눈의 중심에 빨간색 점을 그립니다.
        image = cv2.circle(image, (eyes_center_x_pixel, eyes_center_y_pixel), radius=5, color=(0, 0, 255), thickness=-1)
        
        nose_landmark = face_landmarks[4]
        nose_x = int(nose_landmark.x * image_width)
        nose_y = int(nose_landmark.y * image_height)

        image = cv2.circle(image, (nose_x, nose_y), radius=5, color=(0, 0, 255), thickness=-1)

        if left_eye_landmark.y > right_eye_landmark.y:
            point_3rd = (right_eye_landmark.x, left_eye_landmark.y)
            direction = -1  # rotate image direction to clock
        else:
            point_3rd = (left_eye_landmark.x, right_eye_landmark.y)
            direction = 1  # rotate inverse direction of clock
 

        a = trignometry_for_distance((left_eye_landmark.x,left_eye_landmark.y),
                                     point_3rd)
        b = trignometry_for_distance((right_eye_landmark.x,right_eye_landmark.y),
                                     point_3rd)
        c = trignometry_for_distance((right_eye_landmark.x,right_eye_landmark.y),
                                     (left_eye_landmark.x,left_eye_landmark.y))
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = (np.arccos(cos_a) * 180) / math.pi
 
        if direction == -1:
            angle = 90 - angle
        else:
            angle = -(360-angle)

        # if direction == -1:
        #     angle = 360 - angle
        # else:
        #     angle = -(360-angle)
 
        # rotate image
        image = Image.fromarray(image)
        rotated_image = np.array(image.rotate(direction*angle))
        
        return rotated_image

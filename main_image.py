import cv2
import numpy as np
from typing import Tuple, Union
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt

from run import face_detection,face_landmark



main_img = cv2.imread('test 2.png')
main_img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(main_img,cv2.COLOR_BGR2RGB))


detection_result_FD = face_detection.run_face_detection(main_img_mp)

main_img_copy = np.copy(main_img_mp.numpy_view())
annotated_image = face_detection.visualize(main_img_copy, detection_result_FD)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)



for detection in detection_result_FD.detections:
  bounding_box = detection.bounding_box
  left_top = (bounding_box.origin_x, bounding_box.origin_y)
  right_bottom = (bounding_box.origin_x + bounding_box.width, bounding_box.origin_y + bounding_box.height)
  cropped_image = main_img[int(left_top[1])-200:int(right_bottom[1])+50, int(left_top[0])-50:int(right_bottom[0])+50]
  cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB)

if detection_result_FD.detections:
  cropped_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_image)
  
  detection_result_FL = face_landmark.run_face_landmark(cropped_image_mp)

  # print(face_landmark.asy_val(detection_result_FL))
  # annotated_image = face_landmark.draw_landmarks_on_image(cropped_image_mp.numpy_view(), detection_result_FL)

  rotated_img = face_landmark.align_face(detection_result_FL,cropped_image)
  rotated_img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rotated_img)

  detection_result_RI = face_landmark.run_face_landmark(rotated_img_mp)

  annotated_image = face_landmark.draw_landmarks_on_image(rotated_img_mp.numpy_view(), detection_result_RI)

  asy_angle = face_landmark.asy_val(detection_result_RI,rotated_img)

  cv2.putText(annotated_image, str(asy_angle) , (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

  
  cv2.imshow('result',annotated_image)
  cv2.waitKey(0)
  # cv2.imshow('result',rotated_img)




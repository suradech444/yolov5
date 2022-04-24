from pickle import NONE
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave, imread
from mrcnn.m_rcnn import *

face_detector = dlib.get_frontal_face_detector()

datFile =  "PROJECT/content/shape_predictor_68_face_landmarks.dat"
landmark_detector = dlib.shape_predictor(datFile)

img_path = "PROJECT/testdlib2.jpg"
 
#read with dlib
img = dlib.load_rgb_image(img_path)

faces = face_detector(img, 1)

landmark_tuple = []
for k, d in enumerate(faces):
   landmarks = landmark_detector(img, d)
   for n in range(0, 27):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmark_tuple.append((x, y))
      cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

routes = []
 
for i in range(15, -1, -1):
   from_coordinate = landmark_tuple[i+1]
   to_coordinate = landmark_tuple[i]
   routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[0]
to_coordinate = landmark_tuple[17]
routes.append(from_coordinate)
 
for i in range(17, 20):
   from_coordinate = landmark_tuple[i]
   to_coordinate = landmark_tuple[i+1]
   routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[19]
to_coordinate = landmark_tuple[24]
routes.append(from_coordinate)
 
for i in range(24, 26):
   from_coordinate = landmark_tuple[i]
   to_coordinate = landmark_tuple[i+1]
   routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[26]
to_coordinate = landmark_tuple[16]
routes.append(from_coordinate)
routes.append(to_coordinate)

for i in range(0, len(routes)-1):
   from_coordinate = routes[i]
   to_coordinate = routes[i+1]
   img = cv2.line(img, from_coordinate, to_coordinate, (255, 255, 0), 1)

mask = np.zeros((img.shape[0], img.shape[1]))
mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
mask = mask.astype(np.bool)
 
out = np.zeros_like(img)
out[mask] = img[mask]

imsave('PROJECT/file.jpg',out)

img = cv2.imread('PROJECT/file.jpg', 0)

face_area = cv2.countNonZero(img)

# print("Number of dark pixels:")
# print(face_area)

height, width = img.shape
n_total = height * width

# print("Percentage of dark pixels:")
# print(face_area / n_total * 100)

#---------------------------------------------------------------------------------------------------------------

img = cv2.imread("PROJECT/file.jpg")

# test_model = model.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
# WEIGHTS_PATH = "model_acne.h5"
# test_model.load_weights(WEIGHTS_PATH, by_name=True)

test_model, inference_config = load_inference_model(1, "PROJECT/content/Model/Acne/model_acne.h5")

image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r = test_model.detect([image])[0]

from mrcnn.visualize import random_colors, get_mask_contours, draw_mask

object_count = len(r["class_ids"])
colors = random_colors(object_count)
areaacne = 0
count = 0
for i in range(object_count):
  mask = r["masks"][:, :, i]
  contours = get_mask_contours(mask)
  for cnt in contours:
    #cv2.polylines(img, [cnt], True, colors[i], 2)
    cv2.fillPoly(img, [cnt], (0,255,0))

    img = draw_mask(img, [cnt], (0,255,0))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  
    lower_range = np.array([50, 220, 20])   
    upper_range = np.array([100, 255, 255])
 
    mask = cv2.inRange(hsv, lower_range, upper_range)

    n_black = cv2.countNonZero(mask)
    print(n_black)

    areaacne = areaacne + n_black

    count+=1

plt.imshow(img)
plt.show()

# print(count)

print("Area of pic")
print(n_total)

print("Face area pixels:")
print(face_area)

print("Percentage of face pixels:")
print(face_area / n_total * 100)

print("area acne pixels:")
print(areaacne)

percent = areaacne/face_area *100

print("PERCENT : "'%.2f' %(percent))

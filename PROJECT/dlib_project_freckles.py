import numpy as np
import cv2
from skimage.io import imsave, imread
from pickle import NONE
import dlib
import matplotlib.pyplot as plt

# step 1 - load the model

face_detector = dlib.get_frontal_face_detector()

datFile =  "PROJECT/content/shape_predictor_68_face_landmarks.dat"
landmark_detector = dlib.shape_predictor(datFile)

img_path = "PROJECT/test/3.jpg"
 
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

#-----------------------------------------------------------------------------------------------------------------------------------------------------------

net = cv2.dnn.readNet('w2/best.onnx')

# step 2 - feed a 640x640 image to get predictions

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

image = cv2.imread('PROJECT/file.jpg')
input_image = format_yolov5(image) # making the image square
blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
net.setInput(blob)
predictions = net.forward()

# step 3 - unwrap the predictions to get the object detections 

class_ids = []
confidences = []
boxes = []

output_data = predictions[0]

image_width, image_height, _ = input_image.shape
x_factor = image_width / 640
y_factor =  image_height / 640

for r in range(25200):
    row = output_data[r]
    confidence = row[4]
    if confidence >= 0.4:

        classes_scores = row[5:]
        _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
        class_id = max_indx[1]
        if (classes_scores[class_id] > .25):

            confidences.append(confidence)

            class_ids.append(class_id)

            x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            box = np.array([left, top, width, height])
            boxes.append(box)

class_list = []
with open("PROJECT/text/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()]

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

result_class_ids = []
result_confidences = []
result_boxes = []

for i in indexes:
    result_confidences.append(confidences[i])
    result_class_ids.append(class_ids[i])
    result_boxes.append(boxes[i])

areaacne = 0
count = 0
for i in range(len(result_class_ids)):

    box = result_boxes[i]
    class_id = result_class_ids[i]

    cv2.rectangle(image, box, (0, 255, 0), -1)

    lower_range = np.array([50, 220, 20])   
    upper_range = np.array([100, 255, 255])
 
    mask = cv2.inRange(image, lower_range, upper_range)

    n_black = cv2.countNonZero(mask)
    print(n_black)

    areaacne = areaacne + n_black

    count+=1

    #cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 0), -1)

print(count)

# imsave('PROJECT/mask2.jpg',image)

print("Area of pic")
print(n_total)

print("Face area pixels:")
print(face_area)

print("Percentage of face pixels:")
print(face_area / n_total * 100)

print("area freckles pixels:")
print(areaacne)

percent = areaacne/face_area *100

print("PERCENT : "'%.2f' %(percent))

imsave("misc/kids_detection.png", image)
cv2.imshow("output", image)
cv2.waitKey()
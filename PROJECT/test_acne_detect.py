from mrcnn.m_rcnn import *
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("PROJECT/test/testdlib2.jpg")

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

print(count)
print("Number of dark pixels:")
print(areaacne)
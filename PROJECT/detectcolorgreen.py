import cv2
import numpy as np

img = cv2.imread('testcolor.png')
 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  
lower_range = np.array([50, 220, 20])   
upper_range = np.array([100, 255, 255])
 
mask = cv2.inRange(hsv, lower_range, upper_range)

n_black = cv2.countNonZero(mask)

print("Number of dark pixels:")
print(n_black)

cv2.imshow('image', img)
cv2.imshow('mask', mask)
 
cv2.waitKey(0)
cv2.destroyAllWindows()
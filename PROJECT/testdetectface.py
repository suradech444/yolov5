import cv2

img = cv2.imread('testcolor.png', 0)
img = 255-img # invert image

n_black = cv2.countNonZero(img)

print("Number of dark pixels:")
print(n_black)

height, width = img.shape
n_total = height * width

cv2.imshow("Test",img)

print("Percentage of dark pixels:")
print(n_black / n_total * 100)

cv2.waitKey(0)
cv2.destroyAllWindows()
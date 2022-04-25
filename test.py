import cv2
import torch
from torchvision import models
from skimage.io import imsave, imread
from PIL import Image

model = torch.hub.load('suradech444/yolov5', 'custom', path='w/best.pt')



img = cv2.imread('test/testfk.jpg')
output = model(img)

imsave('PROJECT/file.jpg',output)
print(f'prediction: {output.pred}')
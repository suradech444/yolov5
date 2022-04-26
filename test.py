import cv2
import torch
from torchvision import models
from skimage.io import imsave, imread
from PIL import Image
import matplotlib.pyplot as plt

model = torch.hub.load('', 'custom', path='w1/best.onnx', source='local')

# Image
img = cv2.imread('test/testfk.jpg')

# Inference
results = model(img) # pass the image through our model

results.pandas().xyxy[0]
results.print()

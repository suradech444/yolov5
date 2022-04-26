import cv2
import torch
from torchvision import models
from skimage.io import imsave, imread
from PIL import Image
import matplotlib.pyplot as plt


PATH = 'w/best.pt'
model = torch.hub.load('suradech444/yolov5s', 'yolov5s', pretrained=False)
model.load_state_dict(torch.load(PATH))

# Image
img = cv2.imread('test/testfk.jpg')

# Inference
results = model(img) # pass the image through our model

results.pandas().xyxy[0]
results.print()

import caffe2.python.onnx.backend as backend
import onnx
from IPython.display import Image, display
import torch
import numpy as np

# First load the onnx model
model = onnx.load("animals_caltech.onnx")

# Prepare the backend
rep = backend.prepare(model, device="CPU")

# Transform the image
transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Load and show the image
test_image_name = "giraffe.jpg"
test_image = Image.open(test_image_name)
display(test_image)

# Apply the transformations to the input image and convert it into a tensor
test_image_tensor = transform(test_image)

# Make the input image ready to be input as a batch of size 1
test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

# Convert the tensor to numpy array
np_image = test_image_tensor.numpy()

# Pass the numpy array to run through the ONNX model
outputs = rep.run(np_image.astype(np.float32))

# Dictionary with class name and index
idx_to_class = {0: 'bear', 1: 'chimp', 2: 'giraffe', 3: 'gorilla', 4: 'llama', 5: 'ostrich', 6: 'porcupine', 7: 'skunk', 8: 'triceratops', 9: 'zebra'}

ps = torch.exp(torch.from_numpy(outputs[0]))
topk, topclass = ps.topk(10, dim=1)
for i in range(10):
    print("Prediction", '{:2d}'.format(i+1), ":", '{:11}'.format(idx_to_class[topclass.cpu().numpy()[0][i]]), ", Class Id : ", topclass[0][i].numpy(), " Score: ", topk.cpu().detach().numpy()[0][i])

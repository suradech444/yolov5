import os
from imageai.Detection.Custom import CustomObjectDetection

execution_path = 'model'
input_path = 'static/uploads/'
output_path = 'static/output/'

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , 'detection_model-ex-063--loss-0029.575.h5'))
detector.setJsonPath(os.path.join(execution_path , 'detection_config.json'))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image = os.path.join(input_path, 'acne.jpg'), output_image_path = os.path.join(output_path, 'acne.jpg'))
print('DONE') 

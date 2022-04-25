import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
import cv2
from time import time

class MugDetection:

    def __init__(self, capture_index, model_name):

        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def load_model(self, model_name):

        if model_name:
            model = torch.hub.load('suradech444/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):

        self.model.to(self.device)
        frame = [frame]
        print('ans', frame)
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):

        return self.classes[int(x)]

    def plot_boxes(self, results, frame):

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)
                
        return frame

    def __call__(self):

        frame = cv2.imread("PROJECT/testfk.jpg")

        results = self.score_frame(frame)
        frame = self.plot_boxes(results, frame)

        cv2.imshow('YOLOv5 Detection', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()                                                            

        # while True:
          
        #     ret, frame = cap.read()
        #     assert ret
            
        #     frame = cv2.resize(frame, (416,416))
            
        #     start_time = time()
        #     results = self.score_frame(frame)
        #     frame = self.plot_boxes(results, frame)
            
        #     end_time = time()
        #     fps = 1/np.round(end_time - start_time, 2)
        #     #print(f"Frames Per Second : {fps}")
             
        #     cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
        #     cv2.imshow('YOLOv5 Detection', frame)
 
        #     if cv2.waitKey(5) & 0xFF == 27:
        #         break
      
        # cap.release()
        
        
# Create a new object and execute.
detector = MugDetection(capture_index=0, model_name='best.pt')
detector()
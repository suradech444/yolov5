import streamlit as st
from PIL import Image
from pickle import NONE
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave, imread
from mrcnn.m_rcnn import *

def load_image(image_file):
        
	img = Image.open(image_file)
	return img

def acne_detect():

    try:
        face_detector = dlib.get_frontal_face_detector()
        datFile =  "content/shape_predictor_68_face_landmarks.dat"
        landmark_detector = dlib.shape_predictor(datFile)

        img_path = "imgdetectacne/imgdetect.jpg"

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

        imsave('file.jpg',out)

    except:
        st.error('PICTURE INVALID')

    else:
        st.success("Upload Success")
        try:
            img = cv2.imread('file.jpg', 0)

            face_area = cv2.countNonZero(img)

            height, width = img.shape
            n_total = height * width

            img = cv2.imread("file.jpg")

            test_model, inference_config = load_inference_model(1, "content/Model/Acne/model_acne.h5")

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

            imsave('mask.jpg',mask)

            percent = areaacne/face_area *100

            image_hp = Image.open('emoji/happy.png')
            image_calm = Image.open('emoji/calm.png')
            image_sad = Image.open('emoji/sad.png')

            if percent < 3:
                st.text("Score : GOOD")
                st.image(image_hp, width = 100)

                # print("GOOD")
            
            elif percent > 3 and percent < 5:
                st.text("Score : NORMAL")
                st.image(image_calm, width = 100)
                # print("normal")

            elif percent > 5:
                st.text("Score : BAD")
                st.image(image_sad, width = 100)
                # print("BAD")

            return

        except:
            image_hp = Image.open('emoji/happy.png')

            st.text("Score : GOOD")
            st.image(image_hp, width = 100)

def freckles_detect():
    
    try:
        face_detector = dlib.get_frontal_face_detector()
        datFile =  "content/shape_predictor_68_face_landmarks.dat"
        landmark_detector = dlib.shape_predictor(datFile)

        img_path = "imgdetectfk/imgdetect.jpg"
 
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

        imsave('file.jpg',out)

    except:
        st.error('PICTURE INVALID')
        #print("Invalid")

    else:
        st.success("Upload Success")
        try:
            img = cv2.imread('file.jpg', 0)

            face_area = cv2.countNonZero(img)

            height, width = img.shape
            n_total = height * width

            net = cv2.dnn.readNet('w2/best.onnx')

            def format_yolov5(frame):

                row, col, _ = frame.shape
                _max = max(col, row)
                result = np.zeros((_max, _max, 3), np.uint8)
                result[0:row, 0:col] = frame
                return result

            image = cv2.imread('file.jpg')
            input_image = format_yolov5(image) # making the image square
            blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
            net.setInput(blob)
            predictions = net.forward()

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
            with open("text/classes.txt", "r") as f:
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

                areaacne = areaacne + n_black

                count+=1

            percent = areaacne/face_area *100

            image_hp = Image.open('emoji/happy.png')
            image_calm = Image.open('emoji/calm.png')
            image_sad = Image.open('emoji/sad.png')

            if percent < 3:
                st.text("Score : GOOD")
                st.image(image_hp, width = 100)
                
                # print("GOOD")
            
            elif percent > 3 and percent < 5:
                st.text("Score : NORMAL")
                st.image(image_calm, width = 100)

                # print("normal")

            elif percent > 5:
                st.text("Score : BAD")
                st.image(image_sad, width = 100)

                # print("BAD")

            return

        except:
            image_hp = Image.open('emoji/happy.png')

            st.text("Score : GOOD")
            st.image(image_hp, width = 100)

def melesma_detect():

    try:
        face_detector = dlib.get_frontal_face_detector()
        datFile =  "content/shape_predictor_68_face_landmarks.dat"
        landmark_detector = dlib.shape_predictor(datFile)

        img_path = "imgdetectmelesma/imgdetect.jpg"
 
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

        imsave('file.jpg',out)

    except:
        st.error('PICTURE INVALID')
        #print("Invalid")

    else:
        st.success("Upload Success")
        try:
            img = cv2.imread('file.jpg', 0)

            face_area = cv2.countNonZero(img)

            height, width = img.shape
            n_total = height * width

            img = cv2.imread("file.jpg")

            test_model, inference_config = load_inference_model(1, "content/Model/Melesma/w_melesma.h5")

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

            imsave('mask.jpg',mask)

            percent = areaacne/face_area *100

            image_hp = Image.open('emoji/happy.png')
            image_calm = Image.open('emoji/calm.png')
            image_sad = Image.open('emoji/sad.png')

            if percent < 3:
                st.text("Score : GOOD")
                st.image(image_hp, width = 100)
                # print("GOOD")
            
            elif percent > 3 and percent < 5:
                st.text("Score : NORMAL")
                st.image(image_calm, width = 100)
                # print("normal")

            elif percent > 5:
                st.text("Score : BAD")
                st.image(image_sad, width = 100)
                # print("BAD")

            return

        except:
            image_hp = Image.open('emoji/happy.png')

            st.text("Score : GOOD")
            st.image(image_hp, width = 100)

st.sidebar.header('APPLICATION SKIN TEST BY MOBILE CAMERA')
st.sidebar.header('Choose Prediction Model')

model_choice = st.sidebar.selectbox('Select Prediction Model', ['Acne', 'Freckles', 'Melesma'], key='1')

if model_choice == 'Acne':
    
    st.title("Acne")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    imgname = "imgdetect.jpg"
    if image_file is not None:
        st.image(load_image(image_file),width=300)
        with open(os.path.join("imgdetectacne",imgname),"wb") as f: 
            f.write(image_file.getbuffer())         

    if st.button('Analysis'):
        acne_detect()

elif model_choice == 'Freckles':

    st.title("Freckles")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    imgname = "imgdetect.jpg"
    if image_file is not None:
        st.image(load_image(image_file),width=300)
        with open(os.path.join("imgdetectfk",imgname),"wb") as f: 
            f.write(image_file.getbuffer())         

    if st.button('Analysis'):
        freckles_detect()
    
elif model_choice == 'Melesma':

    st.title("Melesma")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    imgname = "imgdetect.jpg"
    if image_file is not None:
        st.image(load_image(image_file),width=300)
        with open(os.path.join("imgdetectmelesma",imgname),"wb") as f: 
            f.write(image_file.getbuffer())         

    if st.button("Analysis"):
        melesma_detect()

    
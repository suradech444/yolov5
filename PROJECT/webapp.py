import streamlit as st
import streamlit_modal as modal
import streamlit.components.v1 as components
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pickle import NONE
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

                    areaacne = areaacne + n_black

                    count+=1

            imsave('mask.jpg',mask)

            percent = areaacne/face_area *100
            image_hp = Image.open('emoji/happy.png')
            image_calm = Image.open('emoji/calm.png')
            image_sad = Image.open('emoji/sad.png')

            if percent < 3:
                print(count)
                st.write("Detect acne count : "'%d' %(count))
                st.write("Percent acne on face : "'%.2f' %(percent))
                st.text("Score : GOOD")
                st.image(image_hp, width = 100)
                st.markdown("""___""")
                st.title("คำแนะนำ")
                st.write("วิธีที่ 1 - ล้างหน้าให้สะอาด")
                st.write("วิธีที่ 2 - ทานอาหารให้มีประโยชน์")
                st.write("วิธีที่ 3 - หลีกเลี่ยงแสงแดดและรังสียูวี")
                st.write("วิธีที่ 4 - ปกป้องสิวจากมลภาวะ")
                st.write("วิธีที่ 5 - พักผ่อนให้เพียงพอ")
                st.write("วิธีที่ 6 - ใช้ผลิตภัณฑ์ที่ช่วยดูแลปัญหาผิว")

            
            elif percent > 3 and percent < 5:
                st.write("Detect acne count : "'%d' %(count))
                st.write("Percent acne on face : "'%.2f' %(percent))
                st.text("Score : MEDIUM")
                st.image(image_calm, width = 100)
                st.markdown("""___""")
                st.title("คำแนะนำ")
                st.write("วิธีที่ 1 - ล้างหน้าให้สะอาด")
                st.write("วิธีที่ 2 - ทานอาหารให้มีประโยชน์")
                st.write("วิธีที่ 3 - หลีกเลี่ยงแสงแดดและรังสียูวี")
                st.write("วิธีที่ 4 - ปกป้องสิวจากมลภาวะ")
                st.write("วิธีที่ 5 - พักผ่อนให้เพียงพอ")
                st.write("วิธีที่ 6 - ใช้ผลิตภัณฑ์ที่ช่วยดูแลปัญหาผิว")

            elif percent > 5:
                st.write("Detect acne count : "'%d' %(count))
                st.write("Percent acne on face : "'%.2f' %(percent))
                st.text("Score : BAD")
                st.image(image_sad, width = 100)
                st.markdown("""___""")
                st.title("คำแนะนำ")
                st.write("วิธีที่ 1 - ล้างหน้าให้สะอาด")
                st.write("วิธีที่ 2 - ทานอาหารให้มีประโยชน์")
                st.write("วิธีที่ 3 - หลีกเลี่ยงแสงแดดและรังสียูวี")
                st.write("วิธีที่ 4 - ปกป้องสิวจากมลภาวะ")
                st.write("วิธีที่ 5 - พักผ่อนให้เพียงพอ")
                st.write("วิธีที่ 6 - ใช้ผลิตภัณฑ์ที่ช่วยดูแลปัญหาผิว")

        except:
            image_hp = Image.open('emoji/happy.png')
            st.write("Detected acne count : 0")
            st.write("Percent acne on face : 0")
            st.text("Score : GOOD")
            st.image(image_hp, width = 100)
            st.markdown("""___""")
            st.title("คำแนะนำ")
            st.write("วิธีที่ 1 - ล้างหน้าให้สะอาด")
            st.write("วิธีที่ 2 - ทานอาหารให้มีประโยชน์")
            st.write("วิธีที่ 3 - หลีกเลี่ยงแสงแดดและรังสียูวี")
            st.write("วิธีที่ 4 - ปกป้องสิวจากมลภาวะ")
            st.write("วิธีที่ 5 - พักผ่อนให้เพียงพอ")
            st.write("วิธีที่ 6 - ใช้ผลิตภัณฑ์ที่ช่วยดูแลปัญหาผิว")

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
                st.write("Detect freckles count : "'%d' %(count))
                st.write("Percent freckles on face : "'%.2f' %(percent))
                st.text("Score : GOOD")
                st.image(image_hp, width = 100)
                st.markdown("""___""")
                st.title("คำแนะนำ")
                st.write("วิธีที่ 1 - ช้ผลิตภัณฑ์บำรุงผิวที่สามารถลดเลือนกระ ")
                st.write("วิธีที่ 2 - ทำเลเซอร์รักษากระ")
                st.write("วิธีที่ 3 - ใช้ครีมรักษากระ")
                st.write("วิธีที่ 4 - หมั่นผลัดเซลล์ผิวด้วยสมุนไพร และ AHA จากผลไม้")
            
            elif percent > 3 and percent < 5:
                st.write("Detect freckles count : "'%d' %(count))
                st.write("Percent freckles on face : "'%.2f' %(percent))
                st.text("Score : MEDIUM")
                st.image(image_calm, width = 100)
                st.markdown("""___""")
                st.title("คำแนะนำ")
                st.write("วิธีที่ 1 - ช้ผลิตภัณฑ์บำรุงผิวที่สามารถลดเลือนกระ ")
                st.write("วิธีที่ 2 - ทำเลเซอร์รักษากระ")
                st.write("วิธีที่ 3 - ใช้ครีมรักษากระ")
                st.write("วิธีที่ 4 - หมั่นผลัดเซลล์ผิวด้วยสมุนไพร และ AHA จากผลไม้")

            elif percent > 5:
                st.write("Detect freckles count : "'%d' %(count))
                st.write("Percent freckles on face : "'%.2f' %(percent))
                st.text("Score : BAD")
                st.image(image_sad, width = 100)
                st.markdown("""___""")
                st.title("คำแนะนำ")
                st.write("วิธีที่ 1 - ช้ผลิตภัณฑ์บำรุงผิวที่สามารถลดเลือนกระ ")
                st.write("วิธีที่ 2 - ทำเลเซอร์รักษากระ")
                st.write("วิธีที่ 3 - ใช้ครีมรักษากระ")
                st.write("วิธีที่ 4 - หมั่นผลัดเซลล์ผิวด้วยสมุนไพร และ AHA จากผลไม้")

        except:
            image_hp = Image.open('emoji/happy.png')
            st.write("Detected freckles count : 0")
            st.write("Percent freckles of face : 0")
            st.text("Score : GOOD")
            st.image(image_hp, width = 100)
            st.markdown("""___""")
            st.title("คำแนะนำ")
            st.write("วิธีที่ 1 - ช้ผลิตภัณฑ์บำรุงผิวที่สามารถลดเลือนกระ ")
            st.write("วิธีที่ 2 - ทำเลเซอร์รักษากระ")
            st.write("วิธีที่ 3 - ใช้ครีมรักษากระ")
            st.write("วิธีที่ 4 - หมั่นผลัดเซลล์ผิวด้วยสมุนไพร และ AHA จากผลไม้")

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
                st.write("Percent melesma on face : "'%.2f' %(percent))
                st.text("Score : GOOD")
                st.image(image_hp, width = 100)
                st.markdown("""___""")
                st.title("คำแนะนำ")
                st.write("วิธีที่ 1 - การทาครีมรักษาฝ้า")
                st.write("วิธีที่ 2 - การกินวิตามิน A, C, E")
                st.write("วิธีที่ 3 - มาสก์หน้าด้วยหัวไชเท้า")
                st.write("วิธีที่ 4 - มาสก์หน้าด้วยใบบัวบก")
                st.write("วิธีที่ 5 - ทาครีมบำรุงที่มีสารไวท์เทนนิ่ง")
            
            elif percent > 3 and percent < 5:
                st.write("Percent melesma on face : "'%.2f' %(percent))
                st.text("Score : MEDIUM")
                st.image(image_calm, width = 100)
                st.markdown("""___""")
                st.title("คำแนะนำ")
                st.write("วิธีที่ 1 - การทาครีมรักษาฝ้า")
                st.write("วิธีที่ 2 - การกินวิตามิน A, C, E")
                st.write("วิธีที่ 3 - มาสก์หน้าด้วยหัวไชเท้า")
                st.write("วิธีที่ 4 - มาสก์หน้าด้วยใบบัวบก")
                st.write("วิธีที่ 5 - ทาครีมบำรุงที่มีสารไวท์เทนนิ่ง")

            elif percent > 5:
                st.write("Percent melesma on face : "'%.2f' %(percent))
                st.text("Score : BAD")
                st.image(image_sad, width = 100)
                st.markdown("""___""")
                st.title("คำแนะนำ")
                st.write("วิธีที่ 1 - การทาครีมรักษาฝ้า")
                st.write("วิธีที่ 2 - การกินวิตามิน A, C, E")
                st.write("วิธีที่ 3 - มาสก์หน้าด้วยหัวไชเท้า")
                st.write("วิธีที่ 4 - มาสก์หน้าด้วยใบบัวบก")
                st.write("วิธีที่ 5 - ทาครีมบำรุงที่มีสารไวท์เทนนิ่ง")

        except:
            image_hp = Image.open('emoji/happy.png')
            st.write("Percent melesma on face : 0")
            st.text("Score : GOOD")
            st.image(image_hp, width = 100)
            st.markdown("""___""")
            st.title("คำแนะนำ")
            st.write("วิธีที่ 1 - การทาครีมรักษาฝ้า")
            st.write("วิธีที่ 2 - การกินวิตามิน A, C, E")
            st.write("วิธีที่ 3 - มาสก์หน้าด้วยหัวไชเท้า")
            st.write("วิธีที่ 4 - มาสก์หน้าด้วยใบบัวบก")
            st.write("วิธีที่ 5 - ทาครีมบำรุงที่มีสารไวท์เทนนิ่ง")

st.set_page_config(page_title='APPLICATION SKIN TEST BY MOBILE CAMERA')
st.sidebar.header('APPLICATION SKIN TEST BY MOBILE CAMERA')

open_modal = st.sidebar.button("คู่มือการใช้งาน")
if open_modal:
    modal.open()

if modal.is_open():
    with modal.container():
        components.html(
        """
        <style>
            .center {
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 50%;
            }
        </style>
        <div style="width: 700px; height: 800px; overflow-y: scroll; scrollbar-arrow-color:blue; scrollbar-face-color: #e7e7e7; scrollbar-3dlight-color: #a0a0a0; scrollbar-darkshadow-color:#888888">  
        <div>
            <h1>คู่มือการใช้งาน<h1>
            <hr width="100%">
        </div>
        <div>
            <p>ขั้นตอนที่ 1 : ผู้ใช้งานต้องถ่ายภาพใบหน้าตนเอง เป็นภาพหน้าตรง แสงสีขาว</p>
            <img src="https://drive.google.com/uc?id=1AjTQeBQg7iuVaYtZE0N9Cm8peyzWp4Zt" class="center">
        </div>
        <div>
            <p>ขั้นตอนที่ 2 : ในส่วนของการเลือกโมเดล ให้ผู้ใช้งานเลือกโมเดลที่ต้องการตรวจสอบ</p>
            <img src="https://drive.google.com/uc?id=16jjujpA2QFC7eOB7WJ_KOf3j0aOBHK_b" class="center">
        </div>
        <div>
            <p>ขั้นตอนที่ 3 : ผู้ใช้งานนำรูปภาพที่ถ่ายไว้ตามขั้นตอนที่ 1 โดยกดปุ่ม Browse file</p>
            <img src="https://drive.google.com/uc?id=1uAPZw_dUKnFt1GK7dmAr88GB7sWAtKHM" width="690" height="300">
        </div>
        <div>
            <p>ขั้นตอนที่ 4 : ผู้ใช้งานกดปุ่ม Analysis แล้ว Web Application จะทำการวิเคราะห์รูปภาพและสิ่งที่ต้องการตรวจจับ</p>
            <img src="https://drive.google.com/uc?id=1NAjjM6p7qPIGo_CAGTuqEicb2b2JyCYj"width="720" height="550">
            <p>     เมื่อรูปผู้ใช้งานถูกต้องตามเงื่อนไข Web Apllication จะแจ้ง Upload Success</p>
            <img src="https://drive.google.com/uc?id=1Svd0spFryw20Xzah9s7gANHvv-s095-g"width="720" height="550">
            <p>     เมื่อรูปผู้ใช้งานไม่ถูกต้องตามเงื่อนไข Web Apllication จะแจ้ง Picture Invalid</p>
            <img src="https://drive.google.com/uc?id=19ykvntivz3rOWc3Khyo3qsNGaWrE5BMz"width="720" height="550">
        </div>
        <div>
            <p>ขั้นตอนที่ 5 : ทาง Web Application จะขึ้นผลลัพธ์การตรวจจับและคำแนะนำ</p>
            <img src="https://drive.google.com/uc?id=14f9dzVKghMjXcFxwFXBrc-G8N8ncyMLA" class="center">
            <hr width="100%">
        </div>            
        <div>
            <h2>สาเหตุการแจ้งเตือน PICTURE INVALID</h2>
            <p>เนื่องจากรูปที่ผู้ใช้งานนำรูปมาไม่ถูกต้องตามเงื่อนไขที่ตั้งไว้ คือ ต้องใช้ภาพที่ติด คิ้ว ตา จมูก ปาก แก้ม และคาง</p>
            <hr width="100%">
        </div>
        <div>
            <h2>คลิปวิดีโอสาธิตการใช้งาน Web Application</h2>
            <iframe width="560" height="315" src="https://www.youtube.com/embed/V5iVwLMlR18" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </div>
        </div> 
        """,
        height=900,
        )

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

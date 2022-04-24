import cv2

#อ่านภาพ
img = cv2.imread("acne.jpg")

#กำหนดขนาด
imgresize = cv2.resize(img,(600,600))


#วาดข้อความบนภาพ
#puttext (ภาพ , ข้อความ , พิกัดข้อความ (x,y) , front ,ขนาดข้อความ,สี,ความหนา)

cv2.putText(imgresize,"CAT",(150,200),cv2.FONT_HERSHEY_SIMPLEX,4,(255,0,0),cv2.LINE_AA)

cv2.imshow("Output",imgresize)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
recognaizer = cv2.face.LBPHFaceRecognizer_create()
recognaizer.read('trainer.yml')

labels = {"personel_name": 1}
with open("labels.pickle.txt",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
       # cv2.rectangle(img , (x,y), (x+w , y+h),(255,0,0),2)
        roi_gray= gray[y:y+h, x:x+w]
        roi_color= img [y:y+h, x:x+w]
        id_, conf = recognaizer.predict(roi_gray)
        if conf>= 45: # and conf<85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke=1
            cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        img_item='10.png'
        cv2.imwrite(img_item,roi_gray)
        color = (255,0,0)
        stroke=2
        end_cord_x= x + w
        end_cord_y= y + h
        cv2.rectangle(img,(x,y),(end_cord_x,end_cord_y),color,stroke)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(0,255,0),1)

    cv2.imshow('Yakala Jooo',img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



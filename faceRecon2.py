import numpy as np
import cv2
import pickle

fc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
recog=cv2.face.LBPHFaceRecognizer_create()
recog.read("db/trainer.yml")

l={}
with open("db/lables.pickle",'rb') as f:
    o_l=pickle.load(f)
    l={v:k for k,v in o_l.items()}

cap = cv2.VideoCapture(0)

while(True):
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in f:
        roi_gray= gray[y:y+h,x:x+w]
        id_,conf = recog.predict(roi_gray)
        if  conf>=45 and conf<=85:
            print(id_)
            print(l[id_])
            print(conf)
            fnt=cv2.FONT_HERSHEY_SIMPLEX
            n1=l[id_]
            col1=(255,255,255)
            st1=2
            cv2.putText(frame,n1,(x,y),fnt,1,col1,st1,cv2.LINE_AA)

            
        col=(0,0,127)
        st=2
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)

    
    cv2.imshow('FaceRecon',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

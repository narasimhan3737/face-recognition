import numpy as np
import cv2

fc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in f:
        if  x>=200 and x<=250 and y>=150 and y<=200:
            col=(0,255,0)
            j=1
        else:
            col=(0,0,255)
            j=0
        st=2
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(200,150),(200+250,150+250),(255,0,0),st)
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)

    
    cv2.imshow('FaceRecon',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2

fc = cv2.CascadeClassifier('data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while(True):
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in f:
        col=(0,0,127)
        st=2
        endx=x+int(w)
        endy=y+int(h/1.5)
        cv2.rectangle(frame,(x+int(w/4),y+int(h/3)),(endx,endy),col,st)

    
    cv2.imshow('eyeRecon',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

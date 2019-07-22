import numpy as np
import cv2
import time as t

fc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
reye=cv2.CascadeClassifier('data/haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture(0)
ex1=0
ey1=0
ex2=0
ey2=0
while(True):
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(gray,1.3,5)
    rey = reye.detectMultiScale(gray,1.6,15)
    for (x,y,w,h) in f:
        col=(0,0,0)
        st=2000
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(int(endx/20000),int(endy/20000)),col,st)
    for (x,y,w,h) in rey:
        endx=x+w
        endy=y+h
        ex=x+int(w/1.8)
        ey=y+int(h/1.5)
        cv2.circle(frame,(ex,ey),int((endx+endy)/5120),(255,255,200),12)
        cv2.circle(frame,(ex,ey),int((endx+endy)/5120),(255,255,127),6)
        

    
    cv2.imshow('fun',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        p=t.time()
        p=int(p)
        dir_="db/fun/"+str(p)+".jpg"
        cv2.imwrite(dir_, frame)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

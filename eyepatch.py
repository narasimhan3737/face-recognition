import numpy as np
import cv2
import time as t

reye=cv2.CascadeClassifier('data/haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture(0)
ex1=0
ey1=0
ex2=0
ey2=0
while(True):
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rey = reye.detectMultiScale(gray,1.3,12)
    for (x,y,w,h) in rey:
        endx=x+w
        endy=y+h
        ex=x+int(w/1.8)
        ey=y+int(h/1.5)
        cv2.circle(frame,(ex,ey),int((endx+endy)/64),(0,0,0),20)
        

    
    cv2.imshow('fun',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        p=t.time()
        p=int(p)
        dir_="db/fun/"+str(p)+".jpg"
        cv2.imwrite(dir_, frame)
        break

cap.release()
cv2.destroyAllWindows()

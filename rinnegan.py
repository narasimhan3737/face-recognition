import numpy as np
import cv2
import time as t

fc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
leye=cv2.CascadeClassifier('data/haarcascade_lefteye_2splits.xml')
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
    ley = leye.detectMultiScale(gray,1.3,12)
    rey = reye.detectMultiScale(gray,1.3,12)
   # for (x,y,w,h) in f:
   #     col=(0,0,0)
    #    st=2000
     #   endx=x+w
      #  endy=y+h
       # cv2.rectangle(frame,(x,y),(endx,endy),col,st)
    for (x,y,w,h) in ley:
        endx=x+w
        enx=w-x
        eny=h-y
        endy=y+h
        ex=x+int(w/2.1)
        ey=y+int(h/1.5)
        cv2.circle(frame,(ex,ey),int(2),(105,0,55),12)
        cv2.circle(frame,(ex,ey),int(6),(0,0,0),1)
        cv2.circle(frame,(ex,ey),int(2),(0,0,0),2)

    for (x,y,w,h) in rey:
        endx=x+w
        endy=y+h
        ex=x+int(w/1.8)
        ey=y+int(h/1.5)
        cv2.circle(frame,(ex,ey),int(2),(105,0,55),12)
        cv2.circle(frame,(ex,ey),int(6),(0,0,0),1)
        cv2.circle(frame,(ex,ey),int(2),(0,0,0),2)
        

    
    cv2.imshow('fun',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        p=t.time()
        p=int(p)
        dir_="db/fun/"+str(p)+".jpg"
        cv2.imwrite(dir_, frame)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

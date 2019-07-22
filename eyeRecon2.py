import numpy as np
import cv2

leye=cv2.CascadeClassifier('data/haarcascade_lefteye_2splits.xml')#eye
reye=cv2.CascadeClassifier('data/haarcascade_righteye_2splits.xml')#eye

path=input("dir:")
path='db/'+path+'/face.jpg'

frame = cv2.imread(path)

while(True):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ley = leye.detectMultiScale(gray,1.2,7)
    rey = reye.detectMultiScale(gray,1.2,8)
    for (x,y,w,h) in ley:
        col=(0,0,127)
        st=2
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)

    for (x,y,w,h) in rey:
        col=(127,0,0)
        st=2
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)

    
    cv2.imshow('eyeRecon',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

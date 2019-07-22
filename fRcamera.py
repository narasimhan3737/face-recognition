import numpy as np
import cv2
import os
import sys
import time as t

fc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

n=input("name:")
m=input("number:")
m=int(m)
imdir='db/'+n+'/'
flg=0
for r,d,Fs in os.walk('db/'):
    p=os.path.basename(r).lower()
    if(p==n):
        flg=1

if(flg==0):
    os.mkdir(imdir)

cap = cv2.VideoCapture(0)
i=0
while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in f:
        col=(0,0,127)
        st=2
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)
        


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        tm=int(t.time())
        dir_=imdir+str(tm)+".png"
        cv2.imwrite(dir_, gray)
        print(i)
        i+=1
        t.sleep(1)
    if(i==m):
        break

cap.release()
cv2.destroyAllWindows()

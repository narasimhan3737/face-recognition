import numpy as np
import cv2

fc = cv2.CascadeClassifier('data/haarcascade_smile.xml')

path=input("dir:")
path='db/'+path+'/face.jpg'

frame = cv2.imread(path)

gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
f = fc.detectMultiScale(gray,1.3,5)
i=0
for (x,y,w,h) in f:
    if(i==1):
        break
    col=(0,0,127)
    st=2
    endx=x+w
    endy=y+h
    cv2.rectangle(frame,(x,y),(endx,endy),col,st)
    i=i+1

cv2.imshow('smileRecon',frame)

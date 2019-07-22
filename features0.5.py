import numpy as np
import cv2

fc=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')#face
leye=cv2.CascadeClassifier('data/haarcascade_lefteye_2splits.xml')#eye
reye=cv2.CascadeClassifier('data/haarcascade_righteye_2splits.xml')#eye
sml=cv2.CascadeClassifier('data/haarcascade_smile.xml')#smile

cap = cv2.VideoCapture(0)#capture video

while(True):
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#gray conversion for lbph recognizer
    f = fc.detectMultiScale(gray,1.5,5)#detect face and recieve co-ordinates
    ley = leye.detectMultiScale(gray,2,5)#detect eye and recieve co-ordinates
    rey = reye.detectMultiScale(gray,2,5)#detect eye and recieve co-ordinates
    sl = sml.detectMultiScale(gray,3,15)#detect smile and recieve co-ordinates
    for (x,y,w,h) in f:
        col=(0,0,127)#Generate red Color
        st=2#Stroke value
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)#Generates red Rectangle for Face
    for (x,y,w,h) in ley:
        col=(0,127,127)#Generate Green Color
        st=2#Stroke value
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)#Generates Green Rectangle for Eye
    for (x,y,w,h) in rey:
        col=(127,127,0)#Generate Green Color
        st=2#Stroke value
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)#Generates Green Rectangle for Eye
        
    for (x,y,w,h) in sl:
        col=(127,0,0)#Generate Blue Color
        st=2#Stroke value
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)#Generates Blue Rectangle for Smile

    
    cv2.imshow('Features RECOG',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()

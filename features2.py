import numpy as np
import cv2

fc=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')#face
eye=cv2.CascadeClassifier('data/haarcascade_eye.xml')#eye
sml=cv2.CascadeClassifier('data/haarcascade_smile.xml')#smile



cap = cv2.VideoCapture(0)#capture video

while(True):
    
    ret, frame = cap.read()
    frame1 = frame
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#gray conversion for lbph recognizer
    f = fc.detectMultiScale(gray,1.3,5)#detect face and recieve co-ordinates
    ey = eye.detectMultiScale(gray,1.3,5)#detect eye and recieve co-ordinates
    sl = sml.detectMultiScale(gray,3,15)#detect smile and recieve co-ordinates
    for (x,y,w,h) in f:
        col=(0,0,127)#Generate red Color
        st=2#Stroke value
        endx=x+w
        endy=y+h
        frame1 = frame[y:endy,x:endx]
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)#Generates red Rectangle for Face
    for (x,y,w,h) in ey:
        col=(0,127,0)#Generate Green Color
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

for (x,y,w,h) in f:
    endx=x+w
    endy=y+h
    cv2.imwrite("db/face.jpg", gray[y:endy,x:endx])

for (x,y,w,h) in ey:
    endx=x+w
    endy=y+h
    cv2.imwrite("db/eye.jpg", gray[y:endy,x:endx])

for (x,y,w,h) in sl:
    endx=x+w
    endy=y+h
    cv2.imwrite("db/smile.jpg", gray[y:endy,x:endx])

print("Done!!!")
cap.release()
cv2.destroyAllWindows()

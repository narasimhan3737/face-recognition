import numpy as np
import cv2

fc = cv2.CascadeClassifier('data/haarcascade_smile.xml')

cv2.namedWindow('smileRecon')
def nothing(x):
    pass

cv2.createTrackbar('x', 'smileRecon',11,250,nothing)
cv2.createTrackbar('y', 'smileRecon',1,26,nothing)

path=input("dir:")
path='db/'+path+'/face.jpg'

frame = cv2.imread(path)


while(True):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    x = cv2.getTrackbarPos('x','smileRecon')
    x=x/10
    y = cv2.getTrackbarPos('y','smileRecon')
    
    f = fc.detectMultiScale(gray,x,y)
    for (x,y,w,h) in f:
        col=(0,0,127)
        st=2
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(endx,endy),col,st)

    
    cv2.imshow('smileRecon',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

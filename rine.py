import numpy as np
import cv2
from PIL import Image

fc = cv2.CascadeClassifier('data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
rine=cv2.imread('db/rine.png')

while(True):
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in f:
        col=(0,0,127)
        st=2
        endx=x+int(w/2)
        endy=y+int(h/2)
        ex=int(w/2)
        ey=int(h/2)
        size = (ex,ey)
        result =np.array(frame,"uint8")
        #rinne= np.reshape(rine,result[y:endy,x:endx].shape)
        rine=result[y:endy,x:endx].shape
        #print (result[y:endy,x:endx])
        result[y:endy,x:endx]=np.array(rinne,"uint8")
        #print (l)
        #cv2.circle(frame,(endx,endy),int((endx+endy)/64),col,st)

    
    cv2.imshow('eyeRecon',result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

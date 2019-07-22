import numpy as np
import cv2

cap = cv2.VideoCapture(0)#capture video

while(True):
    
    ret, frame = cap.read()
   
    cv2.imshow('Features RECOG',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()

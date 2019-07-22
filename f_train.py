import os
import cv2
import numpy as np
from PIL import Image
import pickle

imdir=os.path.dirname(os.path.abspath('db/')) #location of image directory

fc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml') #front face cascade file is loaded

recog=cv2.face.LBPHFaceRecognizer_create() #create a face recognizer (LBPH)

c_id=0  #current id
l_id={}  #label id
x_t=[] #train data
y_l=[] #corresponding label

for r,d,Fs in os.walk(imdir): #browsing on image directories
    for fl in Fs: 
        if ((fl=="face.jpg") or (fl.endswith(".png"))): # if the found file is face.jpg or a .png file
            p=os.path.join(r,fl) # path of the file
            l=os.path.basename(r).replace(" ","-").lower() #directory name
            #print(l,p)
            if not l in l_id: #if the label is not found in label dictionary
                l_id[l]=c_id #create a new entry
                c_id +=1 #increment current id
            id_=l_id[l] #last id entry
            #print(l_id)
            #x_t.append(p)
            #y_l.append(l)
            pil_img=Image.open(p).convert("L") #convert the found image into grayscale
            size=(500,500)
            fImg= pil_img.resize(size, Image.ANTIALIAS) #resize the image for uniform data
            img_array=np.array(fImg,"uint8") #numpy array for the resized image
            #print(img_array)
            f = fc.detectMultiScale(img_array,1.3,5) # detect the image for face
            for (x,y,w,h) in f: 
                roi=img_array[y:y+h,x:x+w] #region of intrest
                #append both region of intrest & id
                x_t.append(roi)
                y_l.append(id_)

#print(x_t)
#print(y_l)
                
#save the data
                
with open("db/lables.pickle",'wb') as f:
    pickle.dump(l_id,f)

recog.train(x_t, np.array(y_l))
recog.save("db/trainer.yml")

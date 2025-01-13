
import tensorflow as tf
import keras
from keras import layers
import cv2
import os
import numpy as np

import cv2

cap = cv2.VideoCapture(0)          #accesing  the webcam
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #detecting the forntal face
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') #detecing the eyes


image_size=224




def process_frame(img_array, image_size):                      #making the captured frame into (224,224,3) size array
    resized_img_array = cv2.resize(img_array, (image_size, image_size))
    img = np.reshape(resized_img_array, (1, image_size, image_size, 3))
    return img

def predict(img, model):
    result = round(model.predict(img, verbose=1)[0][0])
    return result

model = tf.keras.models.load_model('test_model_1.h5')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 6)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_face_gray = gray[y:y+w, x:x+w]
        roi_face = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_face_gray, 1.2, 6)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            roi_eyes=roi_face[ey:ey+eh,ex:ex+ew]
            
            
            
            img = process_frame(roi_eyes,image_size)
            result = predict(img,model)
            if result==1:
                print("Open eye ")
            else:
                print("close eye")
            
            
            cv2.imshow('Live Cam ',frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





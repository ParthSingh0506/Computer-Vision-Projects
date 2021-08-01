import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

with open('network_json') as json_file :
    json_saved_model = json_file.read()

network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('weights.hdf5')
network_loaded.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuaracy')


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
source = cv2.VideoCapture(0)

label_dict = {0:'Mask',1:'No-Mask'}
color_dict = {0:(0,255,0),1:(0,0,255)}

size=4
while True :
    
    #Capturing Image Frame By Frame
    ret,frame = source.read()
    
    #Flip to act as a mirror
    im = cv2.flip(frame,1//size,)

    # Resize the image to speed up detection

    detections = face_classifier.detectMultiScale(frame,1.3,5)

    #Drawing Rectangle on face
    for (x,y,w,h) in detections :
        
        face_img = frame[y:y+w,x:x+w]
        resized = cv2.resize(frame,(64,64))
        normalized = resized/255.0
        reshaped = np.reshape(normalized,(-1,64,64,3))
        reshaped = np.vstack([reshaped])

    #Performing Predictions

        result = network_loaded.predict(reshaped)
        label = np.argmax(result,axis=1)[0]

        
        cv2.rectangle(frame,(x,y) , (x+w,y+h) , color_dict[label],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(frame,label_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    #Displaying the result
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

source.release()
cv2.destroyAllWindows()
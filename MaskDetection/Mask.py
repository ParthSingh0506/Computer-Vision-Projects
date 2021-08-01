import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D , Flatten ,Dense

training_generator = ImageDataGenerator(rescale=1. /255,
                                        rotation_range=7,
                                        horizontal_flip=True,
                                        zoom_range=0.2)

train_data = training_generator.flow_from_directory('train',
                                                    batch_size=32,
                                                    shuffle=True,
                                                    target_size=(64,64),
                                                    class_mode='categorical')


testing_generator = ImageDataGenerator(rescale=1./255)

test_data = testing_generator.flow_from_directory('test',
                                                  batch_size=1,
                                                  shuffle=False,
                                                  target_size=(64,64),
                                                  class_mode='categorical')

#Model Training

with tf.device('/device:GPU:0') :
            
        mask_model = Sequential()
        
        #Adding 1st conv2d layer
        
        mask_model.add(Conv2D(filters=32, kernel_size=3,activation='relu',input_shape=[64,64,3]))
        mask_model.add(MaxPool2D(pool_size=2))
        
        #Adding 2nd conv2d layer
        
        mask_model.add(Conv2D(filters=32, kernel_size=3,activation='relu',input_shape=[64,64,3]))
        mask_model.add(MaxPool2D(pool_size=2))
        
        #Adding 3rd conv2d layer
        
        mask_model.add(Conv2D(filters=32, kernel_size=3,activation='relu',input_shape=[64,64,3]))
        mask_model.add(MaxPool2D(pool_size=2))
        
        mask_model.add(Flatten())
        
        mask_model.add(Dense(units=577,activation='relu'))
        mask_model.add(Dense(units=577,activation='relu'))
        mask_model.add(Dense(units=577,activation='relu'))
        mask_model.add(Dense(units=2,activation='sigmoid'))
        
        s = mask_model.summary()
        mask_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        
        mask_model.fit(train_data,epochs=50)

result = mask_model.predict(test_data)
result = np.argmax(result,axis=1)

from sklearn.metrics import accuracy_score
ac = accuracy_score(result,test_data.classes)

#Loading the model

model_json = mask_model.to_json()
with open('network_json','w') as json_file :
    json_file.write(model_json)

#Saving the model
from keras.models import save_model
save_model(mask_model,'weights.hdf5')

#Opening the model

with open('network_json') as json_file :
    json_saved_model = json_file.read()

network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('weights.hdf5')
network_loaded.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuaracy')

print("THe Accuracy of the model is :-",ac)
print("-------------------")
print("Output Of the following image:-")

from keras.preprocessing import image

test_image = image.load_img('D:/Programming/DeepLearingProjects/CNN Projects/Mask Detection/augmented_image_7.jpg',target_size=(64,64))
test_image= image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

pred = network_loaded.predict(test_image)
pred = np.argmax(pred)

if pred == 0 :
    print("Mask-Detected") 
else :
    print("No-Mask Detected")
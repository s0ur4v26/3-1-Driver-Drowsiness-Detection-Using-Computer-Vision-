import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet import MobileNet


Datadirectory = "D:\python_codes\mrleyedataset"  ## training dataset
image_types = ["Closed_eyes", "Open_eyes"]  ##_ list of Image types
image_size = 224  ##size of each image

def create_training_data():   
    training_data = []
    for category in image_types:                            #category = open or close eyes
        path = os.path.join(Datadirectory, category)
        img_num = image_types.index(category)               
        print(type(img_num))
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                new_img_array = cv2.resize(backtorgb, (image_size, image_size))
                training_data.append([new_img_array, img_num])
            except:
                pass

    return training_data


training_data = create_training_data()

print("total images are")
print(len(training_data))
X=[]
Y=[]
for feature,label in training_data:
    X.append(feature)
    Y.append(label)

X = np.array(X).reshape(-1,image_size,image_size,3)
Y = np.array(Y)


print(X.shape)
print(Y.shape)


num_class = 1
base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(224,224,3))
model = tf.keras.Sequential()

model.add(base_model)
model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(num_class, activation="sigmoid"))
print(model.summary())


# model compiling
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.1)

history = model.fit(X,Y, epochs=20,validation_split=0.1)

model.save("test_model_1.h5")

# result = model.predict(img)














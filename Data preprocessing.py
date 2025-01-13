import warnings
warnings.filterwarnings("ignore") 

import tensorflow as tf
import keras
from keras import layers
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

tf.get_logger().setLevel("WARNING")

Datadirectory = "D:\python_codes\mrleyedataset"  ## training dataset
image_types = ["Closed_eyes", "Open_eyes"]  ##_ list of Image types
image_size = 224  ##size of each image



# def create_training_data():
#     training_data = []
#     for category in image_types:                            #category = open or close eyes
#         path = os.path.join(Datadirectory, category)
#         img_num = image_types.index(category)
#         print(img_num)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
#                 new_img_array = cv2.resize(backtorgb, (image_size, image_size))
#                 training_data.append([new_img_array, img_num])
#             except:
#                 pass

#     return training_data


# training_data = create_training_data()

# print("total images are")
# print(len(training_data))
# X=[]
# Y=[]
# for feature,label in training_data:
#     X.append(feature)
#     Y.append(label)

# X = np.array(X).reshape(-1,image_size,image_size,3)
# Y = np.array(Y)

# print(X.shape)
# print(Y.shape)

# idx = 41230
# img = X[idx]
# print(f"Label = {Y[idx]}")
# print(X[idx].shape)


# plt.imshow(img)
# plt.show()



img_name = "eye.png"
#img_path = os.path.join(Datadirectory, image_types[1], img_name)

def process_frame(img_path):
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    new_img_array = cv2.resize(backtorgb, (image_size, image_size))
    img = np.reshape(new_img_array, (-1, image_size, image_size, 3))
    return img

def predict(img, model):
    result = round(model.predict(img)[0][0])
    return result


model = tf.keras.models.load_model('D:\python_codes/test_model_1.h5')
img = process_frame(img_path=img_name)
result = predict(img, model)
print(result)





print('____ cheking the image _____')
print(result)
#check_model.summary()





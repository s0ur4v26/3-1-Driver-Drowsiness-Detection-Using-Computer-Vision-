Driver Drowsiness Detection using Computer Vision

Introduction

Driver drowsiness is a major cause of road accidents, and detecting early signs of fatigue can help prevent such incidents. This project implements a driver drowsiness detection system using the MRL Eye Dataset and computer vision techniques. The process includes data preprocessing, model training, and real-time detection using a camera.

Steps for Implementation

1. Data Preprocessing

The dataset (https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset) must be cleaned and prepared before training the model.

Script: Data preprocessing.py

Filter and label the data into categories (e.g., open and closed eyes).

Resize images for compatibility with the model.

Normalize pixel values to enhance model performance.

2. Model Training

A MobileNet model is used for training, but you can customize the architecture as needed. Once trained, the model is saved for future use.

Script: Model Training.py

Steps:

Load the preprocessed dataset.

Define the MobileNet model or customize it.

Train the model and validate its performance.

Save the trained model to a file (e.g., drowsiness_model.h5).

3. Finding the Camera ID

To capture real-time video, identify the available camera ID on your system.

Script: Finding the available cameras.py

Use OpenCV to list connected cameras.

Print or save the camera ID for use in the main detection script.

4. Detecting Faces and Eyes

Face and eye detection are critical for identifying drowsiness. The Haar Cascade algorithm from OpenCV is employed for this purpose.

Script: detecting_faces_eyes.py

Steps:

Load the Haar Cascade classifiers for face and eye detection.

Apply the classifiers to frames captured from the camera.

Output the coordinates of detected faces and eyes.

Useful link: Haar Cascade Face and Eye Detection Tutorial

5. Final Integration

Combine all modules to build the complete drowsiness detection system.

Script: Driver Drowsiness Detection (final integration).py

Steps:

Load the trained model.

Use the camera ID to capture video frames.

Detect faces and eyes in each frame.

Predict whether eyes are open or closed using the trained model.

Trigger an alert if drowsiness is detected (e.g., eyes closed for a prolonged period).

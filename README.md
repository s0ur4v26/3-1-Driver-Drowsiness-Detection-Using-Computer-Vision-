<h1>Block Diagram  Representation of the system</h1>

![image](https://github.com/s0ur4v26/3-1-Driver-Drowsiness-Detection-Using-Computer-Vision-/blob/main/Screenshot%202025-01-13%20231554.png?raw=true)






<h3>1. Data Preprocessing</h3>
<p>The dataset (<a href="https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset">MRL Eye Dataset</a>) must be cleaned and prepared before training the model.</p>
<p><strong>Script:</strong> <code>Data preprocessing.py</code></p>
<ul>
    <li>Filter and label the data into categories (e.g., open and closed eyes).</li>
    <li>Resize images for compatibility with the model.</li>
    <li>Normalize pixel values to enhance model performance.</li>
</ul>

<h3>2. Model Training</h3>
<p>A MobileNet model is used for training, but you can customize the architecture as needed. Once trained, the model is saved for future use.</p>
<p><strong>Script:</strong> <code>Model Training.py</code></p>
<ol>
    <li>Load the preprocessed dataset.</li>
    <li>Define the MobileNet model or customize it.</li>
    <li>Train the model and validate its performance.</li>
    <li>Save the trained model to a file (e.g., <code>drowsiness_model.h5</code>).</li>
</ol>

<h3>3. Finding the Camera ID</h3>
<p>To capture real-time video, identify the available camera ID on your system.</p>
<p><strong>Script:</strong> <code>Finding the available cameras.py</code></p>
<ul>
    <li>Use OpenCV to list connected cameras.</li>
    <li>Print or save the camera ID for use in the main detection script.</li>
</ul>

<h3>4. Detecting Faces and Eyes</h3>
<p>Face and eye detection are critical for identifying drowsiness. The Haar Cascade algorithm from OpenCV is employed for this purpose.</p>
<p><strong>Script:</strong> <code>detecting_faces_eyes.py</code></p>
<ol>
    <li>Load the Haar Cascade classifiers for face and eye detection.</li>
    <li>Apply the classifiers to frames captured from the camera.</li>
    <li>Output the coordinates of detected faces and eyes.</li>
</ol>
<p>Useful link: <a href="https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/">Haar Cascade Face and Eye Detection Tutorial</a></p>

<h3>5. Final Integration</h3>
<p>Combine all modules to build the complete drowsiness detection system.</p>
<p><strong>Script:</strong> <code>Driver Drowsiness Detection (final integration).py</code></p>
<ol>
    <li>Load the trained model.</li>
    <li>Use the camera ID to capture video frames.</li>
    <li>Detect faces and eyes in each frame.</li>
    <li>Predict whether eyes are open or closed using the trained model.</li>
    <li>Trigger an alert if drowsiness is detected (e.g., eyes closed for a prolonged period).</li>
</ol>

<h3>Trained  Model  Link </h3>
<p> https://drive.google.com/file/d/1Z-4r2Tak_az3odGfU4K_qIwOXLfRDdVb/view?usp=sharing</p>


<h1>Result:</h1>

![image](https://github.com/user-attachments/assets/cde6d5ec-fbbc-4407-ab05-7a01979826f4)



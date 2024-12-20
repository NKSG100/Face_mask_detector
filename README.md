🖥️ Face Mask Detection System

📄 Project Description:
This project aims to detect whether a person is wearing a mask or not in images, videos, or real-time streams. It combines face detection and mask classification using Deep Learning and Computer Vision techniques. This system is particularly useful for monitoring public spaces to ensure mask compliance.

🚀 Features:
Detect human faces in images, video streams, or real-time webcam feeds.
Classify detected faces as masked or unmasked.
Provides real-time feedback for ensuring safety protocols.
Utilizes optimized pre-trained and custom models for efficient performance.

⚙️ Workflow:
Face Detection:
Use a pre-trained DNN model to locate human faces in the input (image/video).
Mask Classification:

Pass the detected face regions through a Convolutional Neural Network (CNN) model trained on a dataset of masked and unmasked faces.
The model classifies the faces as:
✅ Masked
❌ Unmasked
Real-Time Output:

Combines face detection and mask classification models to work seamlessly for live video streams.
Visual indicators are displayed on the video feed.

📊 Tech Stack:
The following technologies and libraries were used to build this project:

Component	Technology/Library
Programming Language	Python
Face Detection Model	OpenCV, Pre-trained DNN
Mask Detection Model	TensorFlow, Keras (CNN)
Data Processing	NumPy, scikit-learn
Video Processing	OpenCV


🛠️ Installation & Setup
Follow these steps to set up and run the project:

1️⃣ Clone the Repository
git clone https://github.com/NKSG100/Face_mask_detector
cd face-mask-detection

2️⃣ Install Dependencies
Install all required libraries from the requirements.txt file:

Run This Command-

pip install -r requirements.txt
   
3️⃣ Run the Application
Once the setup is complete, run the main script:

python app.py

📂 Project Structure
face-mask-detection/
-> DNN Model
-> -> Pre downloaded Face Model
-> static
-> -> style.css    #CSS files
-> templates
-> -> index.html   #HTML structure file
-> app.py         #main file
-> mask_model     #trained masked model
-> requirements   #required dependencies

📊 Dataset
The dataset contains images of human faces in two categories:

Masked Faces
Unmasked Faces
The data was preprocessed (resized, normalized) to train the MobileNetV2-based CNN model efficiently.

If you want to use a public dataset:

Kaggle Face Mask Dataset

🎥 Demo
Real-time Demo: Works with webcam input to detect and classify faces.
Screenshots:

No mask Detected:
![Screenshot 2024-12-17 124728](https://github.com/user-attachments/assets/5ef375eb-a02f-4924-86f3-ed0f21bf086c)


Mask Detected:
![Screenshot 2024-12-17 124829](https://github.com/user-attachments/assets/8b2f1211-c478-43a3-bd77-d9a9dae53bdd)

🔍 Results
The CNN model achieved the following performance on the test set:

Metric	Score
Accuracy	98.5%
Precision	97.8%
Recall	98.2%
F1-Score	98.0%

💡 Applications
This system can be used for:

Monitoring public spaces (malls, offices, airports, etc.).
Ensuring mask compliance in real-time.
Assisting safety protocols during pandemics.

📜 License
This project is licensed under the MIT License.

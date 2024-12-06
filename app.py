from flask import Flask, render_template, Response
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import imutils

app = Flask(__name__, template_folder='template')  # Specify the template folder

# Load models
face_model = cv2.dnn.readNet("DNN model/deploy.prototxt", "DNN model/res10_300x300_ssd_iter_140000.caffemodel")
mask_model = load_model("mask_detector_model_3.model")

# Initialize the webcam
vs = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = vs.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=400)

        (height, width) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        face_model.setInput(blob)
        detections = face_model.forward()

        faces = []
        co_ordinates = []
        prediction_values = []

        for i in range(0, detections.shape[2]):
            probability = detections[0, 0, i, 2]
            if probability > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x_start, y_start, x_end, y_end) = box.astype("int")
                (x_start, y_start) = (max(0, x_start), max(0, y_start))
                (x_end, y_end) = (min(width - 1, x_end), min(height - 1, y_end))

                face = frame[y_start:y_end, x_start:x_end]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                co_ordinates.append((x_start, y_start, x_end, y_end))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            prediction_values = mask_model.predict(faces, batch_size=32)

        for (box, prediction_values) in zip(co_ordinates, prediction_values):
            (x_start, y_start, x_end, y_end) = box
            (mask, withoutMask) = prediction_values

            label = "Mask Detected" if mask > withoutMask else "WARNING: No Mask"
            color = (0, 255, 0) if label == "Mask Detected" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Make sure index.html is in the 'template' folder

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    global vs
    if vs.isOpened():
        vs.release()  # Release the webcam
    return "Webcam feed stopped"

if __name__ == '__main__':
    app.run(debug=True)

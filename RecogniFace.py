import cv2
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
import pickle

with open("face_recognition_classifier.pkl", "rb") as file:
    classifier = pickle.load(file)
with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)


embedder = FaceNet()

st.set_page_config(page_title="RecogniFace", layout="centered")

st.title("RecogniFace: Real-time Facial Recognition System")

mode = st.sidebar.radio("Choose Input Mode:", ("Webcam", "Upload Image"))

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def extract_embeddings(image):
    image_array = cv2.resize(image, (160, 160)) 
    image_array = np.expand_dims(image_array, axis=0) 
    embeddings = embedder.embeddings(image_array)
    return embeddings

if mode == "Webcam":
    st.header("Real-Time Webcam Detection")

    webcam_active = st.button("Start Webcam")
    if webcam_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the webcam.")
        else:
            webcam_placeholder = st.empty()
            stop_webcam = st.button("Stop Webcam")
            
            while not stop_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to retrieve frame from webcam.")
                    break

                faces = detect_faces(frame)
                for (x, y, w, h) in faces:
                    face = frame[y:y + h, x:x + w]  
                    embeddings = extract_embeddings(face)  # Get the embeddings
                    prediction = classifier.predict(embeddings)  # Predict the label
                    label = label_encoder.inverse_transform(prediction)[0]

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                webcam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            cap.release()
            st.info("Webcam stopped.")

elif mode == "Upload Image":
    st.header("Image Upload for Face Detection")

    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        faces = detect_faces(image_cv)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = image_cv[y:y + h, x:x + w]  
                embeddings = extract_embeddings(face)  # Get the embeddings
                prediction = classifier.predict(embeddings)  # Predict the label
                label = label_encoder.inverse_transform(prediction)[0]

                cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image_cv, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            result_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            st.image(result_image, caption="Recognized Faces", use_container_width=True)
        else:
            st.warning("No faces detected.")
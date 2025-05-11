import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import PIL
from PIL import Image
import tempfile
import os

# --- App Config ---
st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="wide")

st.title("😊 Real-Time Emotion Detection from Image")
st.caption("Upload a photo with a clear face. The app will detect faces and classify emotions.")

# --- Load Model ---
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.h5", compile=False)

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- Upload File ---
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# --- Load Face Detector ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if uploaded_file is not None:
    # Display the image
    img_pil = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)  # (64, 64, 1)
        roi = np.expand_dims(roi, axis=0)   # (1, 64, 64, 1)

        preds = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(preds)]

        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Detected Emotion(s)", use_column_width=True)
else:
    st.info("Upload an image to begin emotion detection.")

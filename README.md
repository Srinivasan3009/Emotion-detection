# Emotion-detection
**Emotion Detection – Naan Mudhalvan Project**

A simple Streamlit web app that detects human emotions from images using a pre-trained **Mini-XCEPTION** model on the FER-2013 dataset.

## 🚀 Features

- Upload any image with visible faces.
- Detects and classifies facial expressions into:
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- Real-time, fast, and works offline.
- Uses the lightweight Mini-XCEPTION CNN model.

## 📁 Project Structure
emotion_detection_app/
│
├── emotion_app.py               # Main Streamlit app file
├── emotion_model.h5             # Pretrained Mini-XCEPTION model (FER-2013)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── assets/
    └── sample.jpg   
    
## 🧠 Model Info

- **Model**: Mini-XCEPTION
- **Dataset**: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Input**: 64x64 grayscale images
- **Output**: Emotion probabilities
- **Emotions Supported**:
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## 🔧 Technologies Used

- **Python**
- **Streamlit** – Web interface
- **OpenCV** – Face detection & image processing
- **Keras / TensorFlow** – Model loading & prediction
- **NumPy & PIL** – Image manipulation

---

## 🚀 How to Run

### Step 1: Clone the repository or set up your files
  git clone https://github.com/your-username/emotion_detection_app.git
  cd emotion_detection_app
### Step 2:Install dependencies
  pip install -r requirements.txt
### Step 3: Download the model
  import urllib.request
  url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
  urllib.request.urlretrieve(url, "emotion_model.h5")
### Step 4: Run the app
  streamlit run emotion_app.py

 **Features**
Upload any image with one or more faces.
Automatically detects faces and classifies their emotions.
Displays annotated results with emotion labels.
Lightweight and runs locally or on cloud platforms.

**Future Enhancements**
Real-time webcam integration
Support for multiple models (e.g., CNN, Vision Transformers)
Display emotion confidence scores
Docker container for easier deployment

**License**
This project is intended for educational use under the Naan Mudhalvan initiative and is not for commercial deployment.


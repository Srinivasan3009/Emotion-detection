# Multimodal Emotion Detection System

This project implements a **Multimodal Emotion Detection System** that analyzes both facial expressions and audio features from video input to detect emotional states. It combines computer vision, audio signal processing, and simple rule-based classification to predict the overall emotion.

## Features
- **Facial Emotion Recognition**: Uses [DeepFace](https://github.com/serengil/deepface) to analyze sampled video frames.
- **Audio Emotion Recognition**: Extracts and processes audio features (MFCC, ZCR, RMSE, etc.) via [librosa](https://librosa.org/).
- **Multimodal Fusion**: Combines visual and vocal cues to provide robust emotion analysis.
- **Interactive Visualizations**: Displays emotion-related charts using [matplotlib](https://matplotlib.org/).
- **Google Colab Compatibility**: Easily works with video uploads in a Google Colab environment.

## How It Works
1. **Upload a Video File**: Upload a video file to Google Colab.
2. **Facial Emotion Detection**: Extract facial frames and detect the dominant emotion using DeepFace.
3. **Audio Emotion Analysis**: Extract audio from the video and analyze it using spectral features.
4. **Classify Audio Emotion**: Classify the audio emotion based on energy levels and ZCR (Zero Crossing Rate).
5. **Multimodal Fusion**: Combine visual and audio data to compute the final emotion.
6. **Visualize Results**: Generate informative visualizations, including bar charts and audio spectrograms.

## Requirements
- **Python 3.x**
- Libraries:
  - `opencv-python`
  - `librosa`
  - `deepface`
  - `moviepy`
  - `matplotlib`
  - `numpy`
  - `google.colab` (for file uploads in Colab)
  - `IPython.display`

## Getting Started
For the best experience, run this project in Google Colab.

### Steps to Run:
1. Clone or upload the script to Google Colab.
2. Import and run the analysis:
   ```python
   from your_script_name import MultimodalEmotionDetector

   detector = MultimodalEmotionDetector()
   detector.run_analysis()

> ⚠️ Note: DeepFace uses pre-trained models and may require initial downloads.

---

## 📊 Output
Dominant Facial Emotions: Visualized across video frames.

Classified Audio Emotion: Based on extracted audio features.

Final Fused Emotion: Combined results from visual and audio cues.

Visuals: Includes bar charts and audio spectrograms.

## Use Cases
This tool is ideal for research and academic projects in areas such as:

Affective computing

Human-computer interaction

Behavioral analysis
---

## 📄 License

This project is for educational and research purposes. Please check individual library licenses for usage in commercial settings.

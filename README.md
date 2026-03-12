# Skin-Cancer-Diagnosis App Using AI

A mobile-based AI system that detects skin cancer from image input. Built with a CNN model in Python, deployed via Flask on AWS, and integrated with a Flutter mobile frontend.

## 🔍 What It Does

- Users upload skin images using a Flutter app
- A CNN model processes the image and returns a diagnosis
- Backend runs on AWS and returns results in real time

## 🧠 Technologies Used

- **Python**: trained a Convolutional Neural Network (CNN) using TensorFlow
- **Flask**: built a REST API to serve the AI model
- **AWS EC2**: hosted the Flask app for public access
- **Flutter**: mobile app for uploading images and displaying results
- **GitLab**: used for version control and collaboration

## 🚀 How It Works

1. Image is captured or selected on the mobile app
2. The image is sent as a POST request to the Flask API (`/predict`)
3. The backend loads the image, preprocesses it, and runs the CNN
4. Flask returns a diagnosis ("Cancer Detected" or "No Cancer")
5. The mobile app displays the result to the user


## 🖼️ App Demo

<img src="flutter_app/screenshots/result_demo.png" width="300" />


## 🖼️ Model Summary

- Built using TensorFlow (Keras)
- Input: 224x224 RGB images
- Output: Binary classification (1 = Cancer, 0 = Normal)

## 📦 Setup (Backend Only)

```bash
git clone https://github.com/omaramer189/Skin-Cancer-Diagnosis-App-Using-AI.git
cd Skin-Cancer-Diagnosis-App-Using-AI
pip install -r requirements.txt
python app.py


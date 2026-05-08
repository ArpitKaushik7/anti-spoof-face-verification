# Anti-Spoof Face Verification

An AI-powered face liveness and anti-spoofing system designed to prevent proxy attendance and presentation attacks such as printed photos, replay videos, and mobile screen spoof attempts.

## Overview

Traditional face recognition systems can be bypassed using static images or replayed videos. This repository focuses on verifying whether the detected face belongs to a real live person before identity recognition or attendance marking.

The system is designed to integrate with smart attendance platforms, surveillance systems, and secure authentication workflows.

## Key Features

* Real vs Spoof face classification
* Detection of printed photo attacks
* Detection of mobile screen replay attacks
* Live webcam inference
* Blink / facial movement based liveness checks
* API-ready architecture for integration
* Designed for future 5G smart campus attendance systems

## Use Cases

* Proxy-proof attendance systems
* Secure biometric login
* Exam hall verification
* Smart campus surveillance
* Visitor authentication

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe
* FastAPI
* NumPy
* Scikit-learn

## Repository Structure

```text
anti-spoof-face-verification/
│── data/
│   ├── real/
│   └── spoof/
│
│── models/
│── notebooks/
│── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│   └── utils.py
│
│── app/
│── requirements.txt
│── .gitignore
│── README.md
```

## Workflow

Camera Input → Face Detection → Liveness Verification → If Real → Identity Recognition / Attendance

## Planned Model Capabilities

* CNN based spoof detection
* Transfer learning using EfficientNet / MobileNet
* Temporal liveness signals
* Confidence scoring
* Real-time inference pipeline

## Future Integration

This module is being developed as an independent verification engine and can later be integrated into larger attendance or biometric systems.

## Getting Started

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

## Vision

Build a robust and scalable face verification layer that reduces proxy fraud in modern attendance systems.

## Author
Arpit Kaushik

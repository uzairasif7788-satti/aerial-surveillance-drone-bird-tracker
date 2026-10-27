# aerial-surveillance-drone-bird-tracker

Real-time Drone and Bird Detection system powered by YOLOv8 and synced with Firebase Realtime Database.

## Overview
This project is an aerial surveillance system built using YOLOv8 to detect and track drones and birds in real-time. It processes input from local video files, webcams, and mobile IP webcams, and automatically updates detection telemetry directly to Firebase Realtime Database.

## Features
- Real-time detection and classification of drones and birds.
- Automatic blur image removal and dataset balancing via Roboflow.
- Firebase Realtime Database integration for tracking active drone count, status, and timestamps.
- Support for live video streams and mobile IP webcam input via OpenCV.

## Model Weights (best.pt)
Due to file size limits on GitHub, the fine-tuned YOLOv8 model weights file (`best.pt`) is hosted externally on Google Drive:
- **Download Link:** [Download best.pt Model Weights](https://drive.google.com/file/d/1y-7KzAJU2RgUWiVqsSfF4_lw5_aQSI1I/view?usp=sharing)

## Tech Stack
- Python
- Ultralytics YOLOv8
- OpenCV
- Firebase Admin SDK
- Roboflow (Dataset Cleaning & Management)

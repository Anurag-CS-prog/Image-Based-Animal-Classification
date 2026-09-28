# Image Based Animal Type Classification for Cattle and Buffaloes

## Project Overview

This project is an AI-based image classification system that identifies whether an uploaded animal image is **Cattle** or **Buffalo**.

The system uses **Computer Vision, Deep Learning, and Transfer Learning** to analyze animal images and provide a prediction along with its confidence score.

## Features

* Upload an animal image through a web interface
* Classify the image as Cattle or Buffalo
* Display prediction confidence
* Image preview before prediction
* FastAPI-based backend
* Web-based frontend
* Trained deep learning model
* Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix

## Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2
* FastAPI
* HTML
* CSS
* JavaScript
* NumPy
* Pillow
* Scikit-learn

## Model

The project uses **MobileNetV2 Transfer Learning** with ImageNet pretrained weights.

The model performs binary classification:

* Cattle
* Buffalo

Input images are resized to **224 × 224 pixels** before prediction.

## Dataset

The project uses a cattle and buffalo image dataset containing images from different breeds.

The dataset is organized into:

```text
Train
Validation
Test
```

The current classification task focuses on identifying the **animal type**, not the breed.

## Model Performance

The trained model was evaluated on an unseen test dataset.

**Test Accuracy: 91.57%**

Additional evaluation metrics include:

* Precision
* Recall
* F1-score
* Confusion Matrix

## Project Workflow

```text
Animal Image
      ↓
Image Preprocessing
      ↓
Data Augmentation
      ↓
MobileNetV2 Transfer Learning
      ↓
Classification
      ↓
Cattle / Buffalo
      ↓
Confidence Score
      ↓
FastAPI Backend
      ↓
Web Frontend
      ↓
Prediction Result
```

## Project Structure

```text
Image-Based-Animal-Classification/
│
├── dataset/
├── models/
├── src/
│   ├── preprocessing/
│   ├── model/
│   ├── prediction/
│   ├── evaluation/
│   └── utils/
│
├── backend/
├── frontend/
├── uploads/
├── notebooks/
├── tests/
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## How to Run

### 1. Activate the environment

```powershell
conda activate animal-classification
```

### 2. Start the FastAPI server

```powershell
uvicorn backend.main:app --reload
```

### 3. Open the application

Open the following address in a browser:

```text
http://127.0.0.1:8000/app
```

### 4. Upload an animal image

Select an image and click **Predict Animal** to get the classification result.

## Future Scope

Future improvements may include:

* Breed identification
* Dashboard and prediction analytics
* Prediction history
* Improved rejection of non-cattle/non-buffalo images
* Mobile application
* Larger and more diverse datasets
* Model fine-tuning for improved classification performance

## Project Goal

The goal of this project is to provide a fast and consistent method for identifying cattle and buffalo from images, reducing the need for manual visual classification and providing a foundation for future livestock-related applications.

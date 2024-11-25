# Advanced Face Recognition and Bias Mitigation
# GROUP-4

This project focuses on developing a fairness-aware face recognition system capable of predicting age, gender, and race attributes from facial images. By leveraging advanced deep learning models and the FairFace dataset, this project addresses bias and ensures equitable performance across diverse demographic groups.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Models](#models)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Challenges and Future Work](#challenges-and-future-work)
9. [Contributors](#contributors)

## Overview

This project mitigates bias in facial recognition systems by:

- Utilizing the balanced FairFace dataset.
- Building multi-task deep learning models.
- Incorporating fairness-aware methodologies.

## Dataset

The **FairFace** dataset is used to ensure inclusivity, comprising over 100,000 facial images annotated for:

- Seven racial groups
- Multiple age brackets
- Binary gender categories

For computational feasibility, a subset of 8,000 images with an 80-20 train-test split was used. 
Link to Filtered Dataset: https://drive.google.com/drive/folders/1x9S6F6O2iVwEP4sgy5DdrrWWD5ak2gRC?usp=sharing

## Preprocessing

Key preprocessing steps include:

1. Resizing images to 224x224 pixels.
2. Normalizing pixel values to the range [0, 1].
3. Applying data augmentation (flipping, scaling, cropping, etc.).
4. Label encoding using one-hot encoding for categorical variables.

## Models

The project evaluates and fine-tunes several deep learning architectures:

- **EfficientNetB0** (best performance with 85% gender, 74% age, and 76% race accuracy)
- **ResNet50**
- **MobileNetV2**
- **VGG16**
- **InceptionV3**

## Installation

Follow these steps to set up the environment:

1. Create a virtual environment:
   ```bash
    python -m venv venv

2.Activate the virtual environment:

On Linux/Mac:
```bash

source venv/bin/activate
```

On Windows:
```bash
\venv\Scripts\activate
```

3.Install dependencies:
```bash
pip install -r requirements.txt
```

##. **Usage**
Notebooks

a.FaceRecognition_EfficientNetB0.ipynb: Training and evaluation using EfficientNetB0.

b.Face_Recognition_Multi_task_CNN_model.ipynb: Multi-task CNN model for age, gender, and race prediction.

c.Face_recognition_MobileNetV2.ipynb: Implementation of MobileNetV2.

d.Filtered and Encoded Dataset for Balanced Analysis.ipynb: Data preprocessing and balancing.

e.VGG16.ipynb: Training and evaluation using VGG16.

## **Python Application**

app3.py:
A user-friendly application for real-time predictions. Upload an image or video to get predictions for age, gender, and race.

**Steps to Run**
Prepare the dataset (e.g., fairface_filtered_10000.csv).

Open and execute the desired notebook for training or inference.

Use app3.py for real-time attribute detection.


## Results
EfficientNetB0 outperformed other models:

Model	Gender Accuracy	Age Accuracy	Race Accuracy

EfficientNetB0	85%	   74%	            76%

ResNet50	      75%	   53%               61%

MobileNetV2	   74%	   42%	            45%

VGG16	         73%	   40%	            42%

InceptionV3	   65%	   38%	            36%

## **Challenges and Future Work**

Challenges

Class Imbalance: Race predictions suffered from underrepresented classes.

Overfitting: Particularly in age and race predictions.

Future Work

Expand dataset size for better generalization.

Explore advanced architectures like Vision Transformers.

Incorporate advanced bias mitigation strategies.

## **Contributors**
This project was developed by Group 4 as part of the EDS 6397 course in Fall 2024:

Bala Srimani Durga Devi Chikkala

Saikiran Anugam

Vishnu Vardhan Mullapudi

Azharmadani Syed

Sunil Kumar Kommineni






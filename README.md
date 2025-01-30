# COVID-19 X-ray Image Classification

## Objective
This project aims to build a Convolutional Neural Network (CNN) to classify X-ray images into three categories:
1. **COVID-19** - Patients affected by COVID-19.
2. **Viral Pneumonia** - Patients suffering from viral pneumonia, which shares some symptoms with COVID-19.
3. **Normal** - Healthy individuals with no symptoms.

## Dataset Overview
- The dataset contains images converted into NumPy arrays and corresponding labels.
- Data files:
  - `CovidImages.npy` - Contains X-ray images.
  - `CovidLabels.csv` - Corresponding labels.
- The dataset is imbalanced, with COVID-19 cases being the most frequent.

## Prerequisites
Before running the code, install the required dependencies:

```bash
pip install tensorflow==2.15.0 scikit-learn==1.2.2 seaborn==0.13.1 \
matplotlib==3.7.1 numpy==1.25.2 pandas==2.0.3 opencv-python==4.8.0.76 -q --user
```

## Data Preprocessing
- Convert images from BGR to RGB.
- Resize images from `128x128` to `64x64` for computational efficiency.
- Normalize pixel values.
- Encode categorical labels for training.
- Split the dataset into training, validation, and test sets.

## Model Architecture
- Sequential CNN model with:
  - Convolutional layers with ReLU activation.
  - MaxPooling layers for downsampling.
  - Fully connected layers with Dropout for regularization.
  - Softmax activation for multi-class classification.
- Optimizer: Adam.
- Loss function: Categorical Crossentropy.

## Training & Evaluation
- The model is trained for **30 epochs** with a batch size of **32**.
- Training and validation accuracy are visualized using Matplotlib.
- Model performance is assessed with:
  - **Confusion Matrix**
  - **Classification Report**
  - **Accuracy Score**

## Model Improvement
- **Learning Rate Reduction**: ReduceLROnPlateau is used to lower the learning rate when accuracy plateaus.
- **Data Augmentation**: Applied transformations such as rotation and fill modes to improve generalization.

## Results
- The base model performed well with **high accuracy on the test set**.
- Improvements with augmentation and learning rate tuning enhanced generalization.

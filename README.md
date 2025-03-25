# Kubernetes Issue Prediction Model Documentation

## Overview
This project implements a machine learning model to predict potential issues in Kubernetes clusters based on historical data. The model is built using XGBoost and is deployed via a Flask API to provide real-time predictions.

## Dataset
The dataset used for training is `kubernetes_issues.csv`, which contains various system metrics and their corresponding issue labels.
- **Features:** Various metrics related to Kubernetes performance and logs.
- **Target Variable:** `Issue_Label` (Categorical variable indicating the type of issue).

## Model Training
- The dataset is preprocessed by mapping categorical labels to numerical values.
- It is split into training (80%) and testing (20%) subsets.
- An **XGBoost Classifier** is trained with `multi:softmax` objective to handle multiple issue categories.
- Performance is evaluated using **accuracy score** and **classification report**.

## Model Performance
The trained model achieved an accuracy of **99.7%** on the test dataset, with high precision and recall across all categories.

## API Deployment
A Flask-based API is created to serve the model predictions.

### API Endpoints:
1. **GET `/`**
   - Returns a status message: _"Kubernetes Issue Prediction API is running!"_
2. **POST `/predict`**
   - Accepts a JSON payload with feature values.
   - Returns a predicted issue label.

### Example Usage
#### Request:
```json
POST http://127.0.0.1:5000/predict
Content-Type: application/json
{
    "feature1": value1,
    "feature2": value2,
    "feature3": value3
}
```
#### Response:
```json
{
    "prediction": "Issue_Type_1"
}
```

## Future Improvements
- Expand dataset with more real-world Kubernetes logs.
- Fine-tune hyperparameters for better generalization.
- Implement a real-time monitoring dashboard.

## Additional Files
- `kubernetes_model.pkl`: Trained model.
- `main.py`: Codebase for training and API deployment.
- **Presentation File (if applicable)** - Includes a demo and discussion on improvements.

---
For any issues or enhancements, feel free to contribute to the GitHub repository!

# Parkinson's Disease Classification using SVM

This repository contains a machine learning model for classifying Parkinson's disease based on various clinical features using a Support Vector Machine (SVM) classifier. The dataset used is the "Parkinson's Disease Classification" dataset, which includes data collected from patients with Parkinson's disease. The goal is to classify whether a person has Parkinson's disease or not based on various speech-related and other clinical features.

## Files in the Repository

- **parkinsons.data**: This is the dataset containing the features and labels for the classification task. It has 23 columns, where the last column (`status`) is the label (0 for healthy, 1 for Parkinson's disease).
  
- **svm_model.pkl**: The trained SVM model stored using `joblib`. This file is used for making predictions on new data.

- **scaler.pkl**: The `StandardScaler` used for scaling the features. This ensures that any new data can be scaled in the same way as the training data before making predictions.

- **parkinsons_classifier.py**: The Python script that performs the following tasks:
  - Loads the dataset.
  - Prepares the data by removing unnecessary columns and separating features and labels.
  - Splits the data into training and testing sets.
  - Standardizes the features for better performance with SVM.
  - Trains a Support Vector Machine (SVM) classifier.
  - Saves the trained model and scaler using `joblib`.
  - Evaluates the model using accuracy, classification report, and confusion matrix.
  - Makes predictions on new sample data.

## Requirements

The code requires the following Python libraries:
- `pandas`: For data manipulation and loading the dataset.
- `scikit-learn`: For machine learning functions (train-test split, scaling, classification, and model evaluation).
- `joblib`: For saving and loading the trained model and scaler.
- `numpy`: For handling numerical data.
  
To install the required libraries, you can use the following command:

```
pip install -r requirements.txt
```

Here is the `requirements.txt` content:

```
pandas
scikit-learn
joblib
numpy
```

## How to Run the Code

1. **Prepare the Dataset**: Ensure that the `parkinsons.data` dataset is available in the same directory or update the path accordingly.

2. **Run the Python Script**: You can run the `parkinsons_classifier.py` script directly, which will train the model and save the model and scaler to disk.

   ```bash
   python parkinsons_classifier.py
   ```

3. **Output**: After running the script, the model's performance will be printed:
   - **Accuracy**: The percentage of correct predictions on the test set.
   - **Classification Report**: Includes precision, recall, and F1-score for each class (healthy vs. Parkinson's).
   - **Confusion Matrix**: Shows the number of true positives, false positives, true negatives, and false negatives.

4. **Making Predictions with New Data**: After training, you can use the saved model and scaler to predict Parkinson's disease status for new data points. For example, the script includes a sample prediction using a new feature vector (`new_data`).

   ```python
   # Example new data
   new_data = np.array([[0.029, 0.047, -0.008, 0.019, 0.002, -0.015, 0.004, 0.021,
                         0.047, -0.041, -0.006, 0.030, 0.029, -0.029, -0.036, 0.018,
                         0.014, -0.013, 0.004, 0.004, 0.010, 0.011]])  # Ensure exactly 22 features

   # Predict using the saved model
   prediction = model.predict(new_data_scaled)
   print("Predicted class for the new sample:", prediction)
   ```

## Files for Deployment

Once the model is trained, the files `svm_model.pkl` and `scaler.pkl` can be deployed to a production environment where they can be used for making predictions on new data.

To load the trained model and scaler:

```python
import joblib

# Load the model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Use the model and scaler to make predictions
new_data_scaled = scaler.transform(new_data_df)
prediction = model.predict(new_data_scaled)
```

## Conclusion

This code demonstrates how to use a Support Vector Machine (SVM) for classifying Parkinson's disease from clinical data. It shows how to preprocess the data, train the model, and evaluate its performance. Additionally, it explains how to save and reuse the trained model for future predictions.

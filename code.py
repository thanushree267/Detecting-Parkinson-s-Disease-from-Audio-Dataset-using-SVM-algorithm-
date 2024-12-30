import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import numpy as np

# Load the dataset
data = pd.read_csv('parkinsons.data')

# Drop the 'name' column and the label column (adjust the label column name if necessary)
X = data.drop(columns=['name', 'status'])  # Adjust 'status' if the label column has a different name
y = data['status']  # Adjust 'status' if the label column has a different name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for SVMs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save the trained model and the scaler using joblib
joblib.dump(model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Accuracy in percentage
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example new data (Ensure it has exactly 22 features, not 23)
new_data = np.array([[0.029, 0.047, -0.008, 0.019, 0.002, -0.015, 0.004, 0.021,
                      0.047, -0.041, -0.006, 0.030, 0.029, -0.029, -0.036, 0.018,
                      0.014, -0.013, 0.004, 0.004, 0.010, 0.011]])  # Ensure exactly 22 features

# Ensure that new_data is converted into a DataFrame with the same columns as the training data
new_data_df = pd.DataFrame(new_data, columns=X.columns)

# Standardize the new data using the same scaler as for the training data
new_data_scaled = scaler.transform(new_data_df)

# Make prediction for the new sample
prediction = model.predict(new_data_scaled)

# Output the prediction
print("\nPredicted class for the new sample:", prediction)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data and encoder
dataset = pd.read_csv("Dataset_ohe.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.reshape(-1, 1)
encoder = joblib.load('encoder.pkl')
y_encoded = encoder.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
y_test_int = np.argmax(y_test, axis=1)
n_classes = len(encoder.categories_[0])

# Load models
interpreter = tf.lite.Interpreter('cnn_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Corrected CNN prediction function
def cnn_predict(X):
    X_reshaped = X.reshape((-1, X.shape[1], 1)).astype(np.float32)
    predictions = []
    for sample in X_reshaped:
        interpreter.resize_tensor_input(input_details['index'], [1, X.shape[1], 1])
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details['index'], np.expand_dims(sample, axis=0))
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_details['index'])[0])
    return np.array(predictions)

# Get predictions
cnn_probs = cnn_predict(X_test)  # Shape: (n_samples, 8)
svm_preds = svm_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)

# Convert SVM predictions to one-hot
svm_probs = np.zeros((len(svm_preds), n_classes))
svm_probs[np.arange(len(svm_preds)), svm_preds] = 1

# Weighted average (equal weights)
weights = [1/3, 1/3, 1/3]
hybrid_probs = (weights[0] * cnn_probs +
                weights[1] * svm_probs +
                weights[2] * rf_probs)
hybrid_preds = np.argmax(hybrid_probs, axis=1)

# Evaluation metrics
print("\nHybrid Model Metrics:")
print(f"Accuracy: {accuracy_score(y_test_int, hybrid_preds):.4f}")
print(f"Precision: {precision_score(y_test_int, hybrid_preds, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test_int, hybrid_preds, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test_int, hybrid_preds, average='weighted'):.4f}")

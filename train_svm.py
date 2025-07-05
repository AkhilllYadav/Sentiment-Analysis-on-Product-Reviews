import os
import time
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
processed_data_dir = 'processed_data'
model_output_dir = 'models'
model_path = os.path.join(model_output_dir, 'svm_model.joblib')
report_path = os.path.join(model_output_dir, 'svm_report.txt')
cm_path = os.path.join(model_output_dir, 'svm_cm.png')

# Create model output directory if it doesn't exist
os.makedirs(model_output_dir, exist_ok=True)

# --- Load Processed Data ---
print("Loading processed data...")
start_time = time.time()
try:
    X_train_tfidf = joblib.load(os.path.join(processed_data_dir, 'X_train.joblib'))
    X_test_tfidf = joblib.load(os.path.join(processed_data_dir, 'X_test.joblib'))
    y_train = joblib.load(os.path.join(processed_data_dir, 'y_train.joblib'))
    y_test = joblib.load(os.path.join(processed_data_dir, 'y_test.joblib'))
    print(f"Data loaded successfully in {time.time() - start_time:.2f} seconds.")
    print(f"Training features shape: {X_train_tfidf.shape}")
    print(f"Test features shape: {X_test_tfidf.shape}")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure 'feature_engineering.py' ran successfully.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# --- Train SVM Model ---
print("\nTraining SVM model...")
train_start_time = time.time()

# Initialize the model - LinearSVC is faster than SVC for large datasets
# We use CalibratedClassifierCV to get probability estimates
base_svm = LinearSVC(random_state=42, max_iter=1000, C=1.0)
svm_model = CalibratedClassifierCV(base_svm)

# Train the model
svm_model.fit(X_train_tfidf, y_train)
print(f"Model training completed in {time.time() - train_start_time:.2f} seconds.")

# --- Evaluate Model ---
print("\nEvaluating model on the test set...")
eval_start_time = time.time()
y_pred = svm_model.predict(X_test_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)'])
cm = confusion_matrix(y_test, y_pred)

print(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds.")

print("\n--- Evaluation Results ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(cm)

# --- Save Report and Confusion Matrix Plot ---
print(f"\nSaving classification report to: {report_path}")
with open(report_path, 'w') as f:
    f.write(f"SVM Model Evaluation\n")
    f.write("="*30 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))

print(f"Saving confusion matrix plot to: {cm_path}")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative (0)', 'Positive (1)'], yticklabels=['Negative (0)', 'Positive (1)'])
plt.title('Confusion Matrix - SVM')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(cm_path)

# --- Save the Trained Model ---
print(f"\nSaving trained model to: {model_path}")
try:
    joblib.dump(svm_model, model_path)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

end_time = time.time()
print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")
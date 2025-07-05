import os
import time
import pandas as pd
import numpy as np
import joblib
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

# --- Configuration ---
preprocessed_file = os.path.join('data', 'preprocessed_reviews.csv')
output_dir = 'processed_data'
w2v_model_path = os.path.join(output_dir, 'word2vec_model.model')
X_train_w2v_path = os.path.join(output_dir, 'X_train_w2v.joblib')
X_test_w2v_path = os.path.join(output_dir, 'X_test_w2v.joblib')
y_train_w2v_path = os.path.join(output_dir, 'y_train_w2v.joblib')
y_test_w2v_path = os.path.join(output_dir, 'y_test_w2v.joblib')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Load Preprocessed Data ---
print(f"Loading preprocessed data from: {preprocessed_file}")
start_time = time.time()
try:
    df = pd.read_csv(preprocessed_file)
    print(f"Data loaded successfully in {time.time() - start_time:.2f} seconds.")
except FileNotFoundError:
    print(f"Error: File not found at {preprocessed_file}")
    print("Please ensure 'preprocess_data.py' ran successfully.")
    exit()
except Exception as e:
    print(f"An error occurred loading data: {e}")
    exit()

# --- Prepare Text Data for Word2Vec ---
print("Preparing text data for Word2Vec...")
preprocess_start_time = time.time()

# Tokenize the text (split into words)
def tokenize(text):
    if isinstance(text, str):
        return text.lower().split()
    return []

# Apply tokenization
df['tokens'] = df['Text'].apply(tokenize)

print(f"Text tokenization completed in {time.time() - preprocess_start_time:.2f} seconds.")

# --- Train Word2Vec Model ---
print("Training Word2Vec model...")
w2v_start_time = time.time()

# Train Word2Vec model
vector_size = 100  # Dimensionality of the word vectors
window = 5  # Maximum distance between the current and predicted word
min_count = 5  # Ignores all words with total frequency lower than this
workers = 4  # Number of threads to run in parallel

w2v_model = Word2Vec(sentences=df['tokens'], vector_size=vector_size, window=window, min_count=min_count, workers=workers)

print(f"Word2Vec model training completed in {time.time() - w2v_start_time:.2f} seconds.")
print(f"Vocabulary size: {len(w2v_model.wv.key_to_index)}")

# Save the Word2Vec model
w2v_model.save(w2v_model_path)
print(f"Word2Vec model saved to: {w2v_model_path}")

# --- Create Document Vectors ---
print("Creating document vectors from Word2Vec embeddings...")
vector_start_time = time.time()

# Function to create a document vector by averaging word vectors
def document_vector(tokens, model, vector_size):
    # Initialize a zeros vector
    vec = np.zeros(vector_size)
    # Count valid tokens
    count = 0
    # Add vectors for each token if it exists in the model
    for token in tokens:
        if token in model.wv:
            vec += model.wv[token]
            count += 1
    # Average the vectors
    if count > 0:
        vec /= count
    return vec

# Create document vectors for each review
X = np.array([document_vector(tokens, w2v_model, vector_size) for tokens in df['tokens']])
y = df['Sentiment'].values

print(f"Document vectors created in {time.time() - vector_start_time:.2f} seconds.")
print(f"Feature matrix shape: {X.shape}")

# --- Train-Test Split ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# --- Save Processed Data ---
print(f"\nSaving processed Word2Vec data to: {output_dir}")
save_start_time = time.time()
try:
    joblib.dump(X_train, X_train_w2v_path)
    joblib.dump(X_test, X_test_w2v_path)
    joblib.dump(y_train, y_train_w2v_path)
    joblib.dump(y_test, y_test_w2v_path)
    print(f"Files saved successfully in {time.time() - save_start_time:.2f} seconds.")
except Exception as e:
    print(f"Error saving files: {e}")

end_time = time.time()
print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")
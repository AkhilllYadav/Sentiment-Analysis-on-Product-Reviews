import os
import time
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
preprocessed_file = os.path.join('data', 'preprocessed_reviews.csv')
output_dir = 'processed_data'
X_train_bert_path = os.path.join(output_dir, 'X_train_bert.joblib')
X_test_bert_path = os.path.join(output_dir, 'X_test_bert.joblib')
y_train_bert_path = os.path.join(output_dir, 'y_train_bert.joblib')
y_test_bert_path = os.path.join(output_dir, 'y_test_bert.joblib')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Load Preprocessed Data ---
print(f"Loading preprocessed data from: {preprocessed_file}")
start_time = time.time()
try:
    df = pd.read_csv(preprocessed_file)
    # For BERT, we'll use a smaller subset to manage processing time
    sample_size = 10000  # Adjust as needed
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    print(f"Data loaded successfully in {time.time() - start_time:.2f} seconds.")
    print(f"Using a sample of {len(df)} reviews for BERT processing.")
except FileNotFoundError:
    print(f"Error: File not found at {preprocessed_file}")
    print("Please ensure 'preprocess_data.py' ran successfully.")
    exit()
except Exception as e:
    print(f"An error occurred loading data: {e}")
    exit()

# --- Load BERT Model and Tokenizer ---
print("Loading BERT model and tokenizer...")
model_start_time = time.time()

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()  # Set model to evaluation mode

print(f"BERT model loaded in {time.time() - model_start_time:.2f} seconds.")
print(f"Using device: {device}")

# --- Extract BERT Embeddings ---
print("Extracting BERT embeddings...")
extract_start_time = time.time()

# Function to get BERT embeddings for a text
def get_bert_embedding(text, tokenizer, model, device):
    # Truncate text if it's too long (BERT has a limit of 512 tokens)
    if isinstance(text, str):
        # Tokenize and prepare input
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the [CLS] token embedding as the sentence representation
        # This is the first token in the sequence (index 0)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return embedding
    else:
        # Return zeros if text is not a string
        return np.zeros(768)  # BERT base has 768 dimensions

# Extract embeddings for each review
embeddings = []
for text in tqdm(df['Text'], desc="Processing reviews"):
    embedding = get_bert_embedding(text, tokenizer, model, device)
    embeddings.append(embedding)

X = np.array(embeddings)
y = df['Sentiment'].values

print(f"BERT embeddings extracted in {time.time() - extract_start_time:.2f} seconds.")
print(f"Feature matrix shape: {X.shape}")

# --- Train-Test Split ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# --- Save Processed Data ---
print(f"\nSaving processed BERT data to: {output_dir}")
save_start_time = time.time()
try:
    joblib.dump(X_train, X_train_bert_path)
    joblib.dump(X_test, X_test_bert_path)
    joblib.dump(y_train, y_train_bert_path)
    joblib.dump(y_test, y_test_bert_path)
    print(f"Files saved successfully in {time.time() - save_start_time:.2f} seconds.")
except Exception as e:
    print(f"Error saving files: {e}")

end_time = time.time()
print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")
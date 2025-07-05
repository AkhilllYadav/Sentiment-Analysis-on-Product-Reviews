import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Configuration ---
preprocessed_file = os.path.join('data', 'preprocessed_reviews.csv')
model_output_dir = 'models'
model_path = os.path.join(model_output_dir, 'lstm_model.pt')  # Changed to .pt for PyTorch
report_path = os.path.join(model_output_dir, 'lstm_report.txt')
cm_path = os.path.join(model_output_dir, 'lstm_cm.png')
history_path = os.path.join(model_output_dir, 'lstm_history.png')

# Create output directory if it doesn't exist
os.makedirs(model_output_dir, exist_ok=True)

# --- Load Preprocessed Data ---
print(f"Loading preprocessed data from: {preprocessed_file}")
start_time = time.time()
try:
    df = pd.read_csv(preprocessed_file)
    # For deep learning, we'll use a smaller subset to manage training time
    sample_size = 50000  # Adjust as needed
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    print(f"Data loaded successfully in {time.time() - start_time:.2f} seconds.")
    print(f"Using a sample of {len(df)} reviews for LSTM training.")
except FileNotFoundError:
    print(f"Error: File not found at {preprocessed_file}")
    print("Please ensure 'preprocess_data.py' ran successfully.")
    exit()
except Exception as e:
    print(f"An error occurred loading data: {e}")
    exit()

# --- Text Preprocessing for LSTM ---
print("Preprocessing text for LSTM...")
preprocess_start_time = time.time()

# PyTorch-based tokenization
max_features = 10000  # Maximum number of words to keep
max_len = 200  # Maximum sequence length

# Build vocabulary from most common words
all_words = []
for text in df['Text']:
    if isinstance(text, str):
        all_words.extend(text.lower().split())

word_counts = Counter(all_words)
most_common = word_counts.most_common(max_features-1)  # -1 to leave room for <UNK>
vocab = {word: idx+1 for idx, (word, _) in enumerate(most_common)}
vocab['<UNK>'] = len(vocab) + 1  # Add unknown token

# Convert texts to sequences
def text_to_sequence(text, vocab, max_len):
    if not isinstance(text, str):
        return [0] * max_len
    
    words = text.lower().split()
    seq = [vocab.get(word, vocab['<UNK>']) for word in words[:max_len]]
    
    # Pad sequences
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    return seq

# Apply tokenization
X = np.array([text_to_sequence(text, vocab, max_len) for text in df['Text']])
y = df['Sentiment'].values

print(f"Text preprocessing completed in {time.time() - preprocess_start_time:.2f} seconds.")
print(f"Vocabulary size: {len(vocab)}")
print(f"Sequence shape: {X.shape}")

# --- Train-Test Split ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# --- Build LSTM Model ---
print("Building LSTM model...")
model_start_time = time.time()

# Define PyTorch LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1, :, :]
        dense1 = torch.relu(self.fc1(hidden))
        drop = self.dropout(dense1)
        output = torch.sigmoid(self.fc2(drop))
        return output

# Model parameters
embedding_dim = 128
hidden_dim = 128
output_dim = 1
dropout = 0.5
vocab_size = len(vocab) + 2  # +1 for <UNK> and +1 for padding

# Initialize model
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, dropout)
print(model)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# Convert data to PyTorch tensors
X_train_tensor = torch.LongTensor(X_train)
X_test_tensor = torch.LongTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# Create DataLoader for batching
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
print("\nTraining LSTM model...")
train_start_time = time.time()

# For tracking metrics
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# Training loop
epochs = 10
best_val_loss = float('inf')

for epoch in range(epochs):
    # Training
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions, target)
        
        # Calculate accuracy
        predicted_classes = (predictions > 0.5).float()
        correct = (predicted_classes == target).float().sum()
        acc = correct / len(target)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}')
    
    # Calculate average loss and accuracy for the epoch
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    # Validation
    model.eval()
    val_loss = 0
    val_acc = 0
    
    with torch.no_grad():
        # Use a subset of test data for validation
        val_size = min(len(X_test_tensor), 2000)  # Limit validation size for speed
        val_data = X_test_tensor[:val_size]
        val_targets = y_test_tensor[:val_size]
        
        val_predictions = model(val_data)
        v_loss = criterion(val_predictions, val_targets)
        
        # Calculate accuracy
        val_predicted_classes = (val_predictions > 0.5).float()
        val_correct = (val_predicted_classes == val_targets).float().sum()
        v_acc = val_correct / len(val_targets)
        
        val_loss = v_loss.item()
        val_acc = v_acc.item()
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

print(f"Model training completed in {time.time() - train_start_time:.2f} seconds.")

# --- Evaluate Model ---
print("\nEvaluating model on the test set...")
eval_start_time = time.time()

# Load the best model
model.load_state_dict(torch.load(model_path))
model.eval()

# Create test DataLoader for batched processing
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate on test set in batches
all_predictions = []
all_targets = []
test_loss = 0
test_correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        # Forward pass
        predictions = model(data)
        loss = criterion(predictions, target)
        
        # Accumulate loss
        test_loss += loss.item() * len(data)
        
        # Calculate accuracy
        predicted_classes = (predictions > 0.5).float()
        test_correct += (predicted_classes == target).float().sum().item()
        total += len(data)
        
        # Store predictions and targets for metrics
        all_predictions.append(predicted_classes.cpu().numpy())
        all_targets.append(target.cpu().numpy())

# Calculate average loss and accuracy
test_loss /= total
test_acc = test_correct / total

# Combine batched predictions and targets
y_pred = np.vstack(all_predictions).flatten()
y_true = np.vstack(all_targets).flatten()

# Calculate metrics
report = classification_report(y_true, y_pred, target_names=['Negative (0)', 'Positive (1)'])
cm = confusion_matrix(y_true, y_pred)

print(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds.")

print("\n--- Evaluation Results ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(cm)

# --- Save Report and Confusion Matrix Plot ---
print(f"\nSaving classification report to: {report_path}")
with open(report_path, 'w') as f:
    f.write(f"LSTM Model Evaluation\n")
    f.write("="*30 + "\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))

print(f"Saving confusion matrix plot to: {cm_path}")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative (0)', 'Positive (1)'], yticklabels=['Negative (0)', 'Positive (1)'])
plt.title('Confusion Matrix - LSTM')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(cm_path)

# --- Plot Training History ---
print(f"Saving training history plot to: {history_path}")
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(train_accs)
plt.plot(val_accs)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_losses)
plt.plot(val_losses)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig(history_path)

end_time = time.time()
print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")
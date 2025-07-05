import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Update this line
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW  # Import AdamW from torch.optim instead
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Configuration ---
preprocessed_file = os.path.join('data', 'preprocessed_reviews.csv')
model_output_dir = 'models'
model_path = os.path.join(model_output_dir, 'bert_model.pt')
report_path = os.path.join(model_output_dir, 'bert_report.txt')
cm_path = os.path.join(model_output_dir, 'bert_cm.png')
history_path = os.path.join(model_output_dir, 'bert_history.png')

# Create output directory if it doesn't exist
os.makedirs(model_output_dir, exist_ok=True)

# --- Load Preprocessed Data ---
print(f"Loading preprocessed data from: {preprocessed_file}")
start_time = time.time()
try:
    df = pd.read_csv(preprocessed_file)
    # For BERT fine-tuning, we'll use a smaller subset to manage training time
    # Change this line
    sample_size = 2000  # Reduced from 20000
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    print(f"Data loaded successfully in {time.time() - start_time:.2f} seconds.")
    print(f"Using a sample of {len(df)} reviews for BERT fine-tuning.")
except FileNotFoundError:
    print(f"Error: File not found at {preprocessed_file}")
    print("Please ensure 'preprocess_data.py' ran successfully.")
    exit()
except Exception as e:
    print(f"An error occurred loading data: {e}")
    exit()

# --- Prepare Data for BERT ---
print("Preparing data for BERT fine-tuning...")
preprocess_start_time = time.time()

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Text'].values, df['Sentiment'].values, 
    test_size=0.2, random_state=42, stratify=df['Sentiment']
)

print(f"Training set size: {len(train_texts)}")
print(f"Test set size: {len(test_texts)}")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create a custom dataset class for BERT
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] else ""
        label = int(self.labels[idx])
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

batch_size = 16  # Adjust based on your GPU memory
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

print(f"Data preparation completed in {time.time() - preprocess_start_time:.2f} seconds.")

# --- Initialize BERT Model ---
print("Initializing BERT model for sequence classification...")
model_start_time = time.time()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-trained BERT model with a classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,  # Binary classification (0: Negative, 1: Positive)
    output_attentions=False,
    output_hidden_states=False
)

# Move model to the device
model = model.to(device)

print(f"Model initialization completed in {time.time() - model_start_time:.2f} seconds.")

# --- Training Setup ---
# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Number of training epochs
epochs = 4

# Total number of training steps
total_steps = len(train_dataloader) * epochs

# Set up the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# --- Training Function ---
# Add this import at the top
from tqdm import tqdm

# Then modify the train_epoch function
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Add progress bar
    progress_bar = tqdm(enumerate(dataloader), desc="Training batches", total=len(dataloader))
    
    for batch_idx, batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Clear gradients
        model.zero_grad()
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters and learning rate
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        
        total_loss += loss.item() * len(labels)
        total_correct += correct
        total_samples += len(labels)
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item(), "accuracy": correct/len(labels)})
        
        # Save intermediate model every 20 batches (adjusted for smaller dataset)
        if (batch_idx + 1) % 20 == 0:
            intermediate_path = os.path.join(model_output_dir, f'bert_model_intermediate_batch{batch_idx+1}.pt')
            try:
                torch.save(model.state_dict(), intermediate_path)
                print(f"Saved intermediate model to {intermediate_path}")
            except Exception as e:
                print(f"Error saving model to {intermediate_path}: {e}")
                # Continue training without stopping
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

# --- Evaluation Function ---
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            
            total_loss += loss.item() * len(labels)
            total_correct += correct
            total_samples += len(labels)
            
            # Store predictions and labels for classification report
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy, all_preds, all_labels

# --- Training Loop ---
print("\nStarting BERT fine-tuning...")
train_start_time = time.time()

# For tracking metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

best_val_loss = float('inf')

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 30)
    
    # Training phase
    print("Training...")
    epoch_start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, scheduler, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    print(f"Training completed in {time.time() - epoch_start_time:.2f} seconds.")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    
    # Validation phase
    print("\nEvaluating...")
    eval_start_time = time.time()
    val_loss, val_acc, _, _ = evaluate(model, test_dataloader, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds.")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

print(f"\nTraining completed in {time.time() - train_start_time:.2f} seconds.")

# --- Final Evaluation ---
print("\nPerforming final evaluation on the test set...")
eval_start_time = time.time()

# Load the best model
model.load_state_dict(torch.load(model_path))

# Evaluate on the test set
test_loss, test_acc, test_preds, test_labels = evaluate(model, test_dataloader, device)

print(f"Final evaluation completed in {time.time() - eval_start_time:.2f} seconds.")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Generate classification report
report = classification_report(
    test_labels, test_preds,
    target_names=['Negative (0)', 'Positive (1)'],
    digits=4
)

# Generate confusion matrix
cm = confusion_matrix(test_labels, test_preds)

# --- Save Results ---
print(f"\nSaving evaluation results to: {report_path}")

# Save classification report
with open(report_path, 'w') as f:
    f.write("BERT Fine-tuned Model Evaluation\n")
    f.write("==============================\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(cm_path)

# Plot and save training history
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(val_accuracies, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(history_path)

print(f"Results saved to {report_path} and {cm_path}")
print(f"Training history plot saved to {history_path}")

end_time = time.time()
print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")
import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your test dataset and model
model_path = 'models/bert_model_intermediate_batch80.pt'
preprocessed_file = os.path.join('data', 'preprocessed_reviews.csv')
model_output_dir = 'models'
report_path = os.path.join(model_output_dir, 'bert_report.txt')
cm_path = os.path.join(model_output_dir, 'bert_cm.png')

# Load the same test data you used for training
df = pd.read_csv(preprocessed_file)
sample_size = 2000
df = df.sample(n=min(sample_size, len(df)), random_state=42)

# Use the same test set as in your training script
from sklearn.model_selection import train_test_split
_, test_texts, _, test_labels = train_test_split(
    df['Text'].values, df['Sentiment'].values, 
    test_size=0.2, random_state=42, stratify=df['Sentiment']
)

# Create the same dataset class
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

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Load the saved model state
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Create test dataset and dataloader
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

# Evaluate the model
print("Evaluating BERT model...")
test_preds, test_labels = evaluate(model, test_dataloader, device)

# Generate classification report
report = classification_report(
    test_labels, test_preds,
    target_names=['Negative (0)', 'Positive (1)'],
    digits=4
)

# Generate confusion matrix
cm = confusion_matrix(test_labels, test_preds)

# Save results
print(f"Saving evaluation results to: {report_path}")
with open(report_path, 'w') as f:
    f.write("BERT Intermediate Model Evaluation (Batch 80)\n")
    f.write("==============================\n")
    f.write(f"Test Accuracy: {(test_preds == test_labels).mean():.4f}\n\n")
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

print(f"Results saved to {report_path} and {cm_path}")
print("Evaluation complete!")
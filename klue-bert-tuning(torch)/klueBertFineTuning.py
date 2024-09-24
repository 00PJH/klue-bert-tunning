import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np
import os

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the CSV data
data_path = 'concat_titles.csv'  # Path to your uploaded file
df = pd.read_csv(data_path)

# Assuming the CSV file has 'text' and 'label' columns for features and labels
texts = df['title'].tolist()
labels = df['label'].tolist()

# Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
model = BertForSequenceClassification.from_pretrained('klue/bert-base', num_labels=2)
model.to(device)

# Tokenization and Encoding
encoded_data = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']

# Split the dataset
train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = train_test_split(
    input_ids, labels, attention_masks, test_size=0.2, random_state=42
)

# Create Dataset class for PyTorch DataLoader
class TextDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = TextDataset(train_inputs, train_masks, train_labels)
val_dataset = TextDataset(val_inputs, val_masks, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training and Evaluation Loop
epochs = 5
training_stats = []
best_val_accuracy = 0.0  # Track best validation accuracy
best_epoch = 0  # To log which epoch had the best model

for epoch in range(epochs):
    start_time = time.time()

    # Training
    model.train()
    total_loss = 0
    correct_preds, total_preds = 0, 0

    for batch in train_loader:
        optimizer.zero_grad()

        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['label'].to(device)

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels
        )
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        preds = torch.argmax(outputs.logits, dim=1)
        correct_preds += (preds == batch_labels).sum().item()
        total_preds += batch_labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct_preds / total_preds  # Track training accuracy
    training_time = time.time() - start_time

    # Validation
    model.eval()
    val_loss = 0
    correct_preds, total_preds = 0, 0
    start_val_time = time.time()

    with torch.no_grad():
        for batch in val_loader:
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels
            )
            loss = outputs.loss
            val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct_preds += (preds == batch_labels).sum().item()
            total_preds += batch_labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_preds / total_preds
    validation_time = time.time() - start_val_time

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        output_dir = "klue-bert-classificationTitles-best-model"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Log stats
    training_stats.append({
        'epoch': epoch + 1,
        'Training Loss': avg_train_loss,
        'Train Accur.': train_accuracy,
        'Valid. Loss': avg_val_loss,
        'Valid. Accur.': val_accuracy,
        'Training Time': training_time,
        'Validation Time': validation_time,
        'Best Model Saved': best_epoch if val_accuracy == best_val_accuracy else ''
    })

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

# Save training stats to CSV
stats_df = pd.DataFrame(training_stats)
stats_df.to_csv("training_stats.csv", index=False)

# Report best epoch
print(f"Best model saved at epoch: {best_epoch} with validation accuracy: {best_val_accuracy:.4f}")

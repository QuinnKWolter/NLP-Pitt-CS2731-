import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import pickle

df = pd.read_csv('combined_training_data.csv')  # The cleaned training data
print("Data loaded successfully.")

df = df[(df['majortopic'] != 'Unknown') & (df['subtopic_code'] != 'Unknown')]  # Remove 'Unknown' labels
print(f"Data filtered: {len(df)} records remaining.")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }

# Device Configuration (Use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer loaded.")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(df['subtopic_code'])))
model.to(device)  # Move model to GPU if available
print("Model loaded and moved to device.")

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['majortopic_code'].astype(str) + '-' + df['subtopic_code'].astype(str))
print("Labels encoded.")

train_texts = df['text'].values
train_labels = df['label'].values

train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Increase batch size if memory permits
print("DataLoader created.")

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
print("Optimizer initialized.")

# Enable Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()
print("Mixed precision training enabled.")

# Training Loop
epochs = 3
for epoch in range(epochs):
    total_loss = 0
    model.train()
    print(f"Starting epoch {epoch + 1}/{epochs}.")

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast():  # Enable mixed precision
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

# Save the model and tokenizer
model.save_pretrained('bert_topic_classifier')
tokenizer.save_pretrained('bert_topic_classifier')
print("Model and tokenizer saved.")

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Label encoder saved.")

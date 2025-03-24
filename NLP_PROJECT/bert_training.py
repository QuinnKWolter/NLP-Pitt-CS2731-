import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import pickle


df = pd.read_csv('combined_training_data.csv')  # The cleaned training data

df = df[(df['majortopic'] != 'Unknown') & (df['subtopic_code'] != 'Unknown')]  # Remove 'Unknown' labels

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


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode labels using majortopic_code and subtopic_code
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['majortopic_code'].astype(str) + '-' + df['subtopic_code'].astype(str))

train_texts = df['text'].values
train_labels = df['label'].values

train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(train_labels)))
model.train()

optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 3
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')


# Save the model
model.save_pretrained('bert_topic_classifier')
tokenizer.save_pretrained('bert_topic_classifier')


# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

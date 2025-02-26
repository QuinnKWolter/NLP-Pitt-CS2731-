import pandas as pd
import numpy as np
import re
import nltk
import string
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import joblib

# Ensure necessary resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("diplomacy_cv.csv")

# Data Preprocessing
def clean_text(text):
    """ Tokenizes, lowercases, removes punctuation, and stopwords. """
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Feature Extraction
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(df['clean_text'])
joblib.dump(vectorizer_bow, 'vectorizer_bow.pkl')  # Save the BoW vectorizer

vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(df['clean_text'])
joblib.dump(vectorizer_tfidf, 'vectorizer_tfidf.pkl')  # Save the TF-IDF vectorizer

# Target Variable
y = df['intent']

# Cross-Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(X, y):
    """ Performs 5-fold cross-validation on the dataset and returns the trained model. """
    model = LogisticRegression()
    accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy').mean()
    y_pred = cross_val_predict(model, X, y, cv=skf)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    return model, (accuracy, precision, recall, f1)

# Train and Save Logistic Regression Models
bow_model, bow_results = evaluate_model(X_bow, y)
tfidf_model, tfidf_results = evaluate_model(X_tfidf, y)

# Fit the models on the entire dataset
bow_model.fit(X_bow, y)
tfidf_model.fit(X_tfidf, y)

joblib.dump(bow_model, 'logistic_bow_model.pkl')
joblib.dump(tfidf_model, 'logistic_tfidf_model.pkl')

# Print Results
print("Logistic Regression BoW Results:", bow_results)
print("Logistic Regression TF-IDF Results:", tfidf_results)

# Neural Network Approach
class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer):
        self.X = torch.tensor(vectorizer.transform(texts).toarray(), dtype=torch.float32)
        self.y = torch.tensor(labels.values, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Train and Evaluate Neural Network
dataset = TextDataset(df['clean_text'], y, vectorizer_tfidf)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = NeuralNet(X_tfidf.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Evaluate Neural Model
y_pred_nn = np.argmax([model(torch.tensor(x, dtype=torch.float32)).detach().numpy() for x in X_tfidf.toarray()], axis=1)
accuracy_nn = accuracy_score(y, y_pred_nn)
precision_nn, recall_nn, f1_nn, _ = precision_recall_fscore_support(y, y_pred_nn, average='binary')

# Print Results
print("Neural Network Results:", (accuracy_nn, precision_nn, recall_nn, f1_nn))

# Save Neural Network Model
torch.save(model.state_dict(), 'neural_net_model.pth')

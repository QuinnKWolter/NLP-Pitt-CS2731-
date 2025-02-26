import pandas as pd
import numpy as np
import torch
import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Ensure necessary resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Load test dataset
df_test = pd.read_csv("diplomacy_test.csv")

# Check for NaN values in 'intent' column and handle them
if df_test['intent'].isnull().any():
    df_test = df_test.dropna(subset=['intent'])  # Drop rows with NaN in 'intent'

# Preprocess the test data
def clean_text(text):
    """ Tokenizes, lowercases, removes punctuation, and stopwords. """
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df_test['clean_text'] = df_test['text'].apply(clean_text)

# Load vectorizers
vectorizer_bow = joblib.load('vectorizer_bow.pkl')  # Load the saved BoW vectorizer
vectorizer_tfidf = joblib.load('vectorizer_tfidf.pkl')  # Load the saved TF-IDF vectorizer

# Transform test data
X_test_bow = vectorizer_bow.transform(df_test['clean_text'])
X_test_tfidf = vectorizer_tfidf.transform(df_test['clean_text'])

# Load Logistic Regression Models
logistic_bow_model = joblib.load('logistic_bow_model.pkl')
logistic_tfidf_model = joblib.load('logistic_tfidf_model.pkl')

# Evaluate Logistic Regression Models
y_test = df_test['intent']

y_pred_bow = logistic_bow_model.predict(X_test_bow)
y_pred_tfidf = logistic_tfidf_model.predict(X_test_tfidf)

accuracy_bow = accuracy_score(y_test, y_pred_bow)
precision_bow, recall_bow, f1_bow, _ = precision_recall_fscore_support(y_test, y_pred_bow, average='binary')

accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
precision_tfidf, recall_tfidf, f1_tfidf, _ = precision_recall_fscore_support(y_test, y_pred_tfidf, average='binary')

print("Logistic Regression BoW Results:", (accuracy_bow, precision_bow, recall_bow, f1_bow))
print("Logistic Regression TF-IDF Results:", (accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf))

# Define Neural Network Model
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

# Load Neural Network Model
model = NeuralNet(X_test_tfidf.shape[1])
model.load_state_dict(torch.load('neural_net_model.pth'))
model.eval()

# Prepare test dataset for neural network
class TextDataset(Dataset):
    def __init__(self, texts, vectorizer):
        self.X = torch.tensor(vectorizer.transform(texts).toarray(), dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

test_dataset = TextDataset(df_test['clean_text'], vectorizer_tfidf)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Predict intents using Neural Network Model
y_pred_nn = []
with torch.no_grad():
    for batch_X in test_dataloader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        y_pred_nn.extend(predicted.numpy())

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'id': df_test['id'],
    'intent_bow': y_pred_bow,
    'intent_tfidf': y_pred_tfidf,
    'intent_nn': y_pred_nn
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

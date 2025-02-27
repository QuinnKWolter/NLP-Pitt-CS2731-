import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import joblib
import argparse
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from qkw3_shared import clean_text, NeuralNet

# Prepare parser, import data.
# (By default, we're using the diplomacy_cv.csv file)

parser = argparse.ArgumentParser(description='Train deception classification models')
parser.add_argument('--filename', type=str, default='diplomacy_cv.csv', help='Input CSV file')
args = parser.parse_args()

df = pd.read_csv(args.filename)

df['clean_text'] = df['text'].apply(clean_text)

# Extract Features
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(df['clean_text'])
joblib.dump(vectorizer_bow, 'vectorizer_bow.pkl')

vectorizer_char = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=5000)
X_char = vectorizer_char.fit_transform(df['clean_text'])
joblib.dump(vectorizer_char, 'vectorizer_char.pkl')

# Target field!
y = df['intent']

# Cross-Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(X, y, model_name, use_feature_selection=False, save_results=True):
    """Performs 5-fold cross-validation and prints performance metrics."""
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    
    if use_feature_selection:
        pipeline = Pipeline([
            ('select', SelectKBest(chi2, k=1000)),  
            ('classifier', model)
        ])
        X = SelectKBest(chi2, k=1000).fit_transform(X, y)  # Apply feature selection inside CV!
    else:
        pipeline = model
    
    accuracy = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy').mean()
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')

    if save_results:
        return model, (accuracy, precision, recall, f1)
    else:
        return model, (accuracy, precision, recall, f1)

print("[>] Training Logistic Regression Models...")
bow_model, bow_results = evaluate_model(X_bow, y, "BoW Unigram", save_results=False)
bow_selected_model, bow_selected_results = evaluate_model(X_bow, y, "BoW Selected", use_feature_selection=True, save_results=False)
char_selected_model, char_selected_results = evaluate_model(X_char, y, "Char-Level TF-IDF", save_results=False)

bow_model.fit(X_bow, y)
bow_selected_model.fit(SelectKBest(chi2, k=1000).fit_transform(X_bow, y), y)
char_selected_model.fit(X_char, y)

print("\n[>] Training Neural Network...")
feature_selector = SelectKBest(chi2, k=1000)
X_bow_selected = feature_selector.fit_transform(X_bow, y)

class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer, feature_selector):
        bow_features = vectorizer.transform(texts)
        selected_features = feature_selector.transform(bow_features)
        self.X = torch.tensor(selected_features.toarray(), dtype=torch.float32)
        self.y = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Neural Network training
dataset = TextDataset(df['clean_text'], y, vectorizer_bow, feature_selector)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

nn_model = NeuralNet(X_bow_selected.shape[1])
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0])) #TODO ought to adjust these weights
optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

# Training loop with validation
best_loss = float('inf')
patience_counter = 0
for epoch in range(20):  # TODO TODO TODO: Keep fiddling with this.
    nn_model.train()
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = nn_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)
    
    # Early stopping. TODO TODO TODO: Keep fiddling with this.
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(nn_model.state_dict(), 'neural_net_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= 5:  # Stop if no improvement for 5 epochs!
            break

# Evaluate my Neural Model
nn_model.eval()
with torch.no_grad():
    # Convert sparse matrix to dense array before creating tensor, to fix a wonky error...
    X_tensor = torch.tensor(X_bow_selected.toarray(), dtype=torch.float32)
    outputs = nn_model(X_tensor)
    y_pred_nn = outputs.argmax(dim=1).numpy()

accuracy_nn = accuracy_score(y, y_pred_nn)
precision_nn, recall_nn, f1_nn, _ = precision_recall_fscore_support(y, y_pred_nn, average='binary')

print("\n[>] Saving Models...")
joblib.dump(bow_model, 'logistic_bow_model.pkl')
joblib.dump(bow_selected_model, 'logistic_bow_selected_model.pkl')
joblib.dump(char_selected_model, 'logistic_char_selected_model.pkl')
torch.save(nn_model.state_dict(), 'neural_net_model.pth')

print("\n[!] Model Performance Results:")
print("\n[*] Logistic Regression - Bag of Words Unigram:")
print(f"[>]   Accuracy: {bow_results[0]:.4f}")
print(f"[>]   Precision: {bow_results[1]:.4f}")
print(f"[>]   Recall: {bow_results[2]:.4f}")
print(f"[>]   F1-Score: {bow_results[3]:.4f}")

print("\n[*] Logistic Regression - Bag of Words with Selected Features:")
print(f"[>]   Accuracy: {bow_selected_results[0]:.4f}")
print(f"[>]   Precision: {bow_selected_results[1]:.4f}")
print(f"[>]   Recall: {bow_selected_results[2]:.4f}")
print(f"[>]   F1-Score: {bow_selected_results[3]:.4f}")

print("\n[*] Logistic Regression - Char-Level TF-IDF:")
print(f"[>]   Accuracy: {char_selected_results[0]:.4f}")
print(f"[>]   Precision: {char_selected_results[1]:.4f}")
print(f"[>]   Recall: {char_selected_results[2]:.4f}")
print(f"[>]   F1-Score: {char_selected_results[3]:.4f}")

print("\n[*] Neural Network Results:")
print(f"[>]   Accuracy: {accuracy_nn:.4f}")
print(f"[>]   Precision: {precision_nn:.4f}")
print(f"[>]   Recall: {recall_nn:.4f}")
print(f"[>]   F1-Score: {f1_nn:.4f}")

print("\n[!] Final Model Performance Summary:")
performance_table = pd.DataFrame({
    "Model": ["BoW Unigram", "BoW Selected", "Char-Level TF-IDF", "Neural Network"],
    "Accuracy": [bow_results[0], bow_selected_results[0], char_selected_results[0], accuracy_nn],
    "Precision": [bow_results[1], bow_selected_results[1], char_selected_results[1], precision_nn],
    "Recall": [bow_results[2], bow_selected_results[2], char_selected_results[2], recall_nn],
    "F1-Score": [bow_results[3], bow_selected_results[3], char_selected_results[3], f1_nn]
})
print(performance_table)

# Confusion matricces
# def plot_confusion_matrix(y_true, y_pred, title):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Truth', 'Lie'],
#                 yticklabels=['Truth', 'Lie'])
#     plt.title(f'Confusion Matrix: {title}')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
#     plt.close()

# Get our predictions
y_pred_bow = cross_val_predict(bow_model, X_bow, y, cv=skf)
y_pred_bow_selected = cross_val_predict(bow_selected_model, SelectKBest(chi2, k=1000).fit_transform(X_bow, y), y, cv=skf)
y_pred_char = cross_val_predict(char_selected_model, X_char, y, cv=skf)

# Plot confusion matrices
# plot_confusion_matrix(y, y_pred_bow, "BoW Unigram")
# plot_confusion_matrix(y, y_pred_bow_selected, "BoW Selected")
# plot_confusion_matrix(y, y_pred_char, "Char-Level TF-IDF")
# plot_confusion_matrix(y, y_pred_nn, "Neural Network")

# Error Analysis
def analyze_errors(texts, y_true, y_pred, model_name):
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
    
    print(f"\n[!] Error Analysis for {model_name}")
    print("\n[x] False Positive Examples (Predicted lie, actually truth):")
    for idx in false_positives[:3]:
        print(f"[>] Text: {texts[idx][:100]}...")
    
    print("\n[x] False Negative Examples (Predicted truth, actually lie):")
    for idx in false_negatives[:3]:
        print(f"[>] Text: {texts[idx][:100]}...")

analyze_errors(df['text'], y, y_pred_bow, "BoW Unigram")
analyze_errors(df['text'], y, y_pred_bow_selected, "BoW Selected")
analyze_errors(df['text'], y, y_pred_char, "Char-Level TF-IDF")
analyze_errors(df['text'], y, y_pred_nn, "Neural Network")

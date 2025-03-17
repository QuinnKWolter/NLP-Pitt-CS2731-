import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.feature_selection import SelectKBest, chi2
import argparse
from qkw3_shared import clean_text, NeuralNet

parser = argparse.ArgumentParser(description='Generate predictions for diplomacy test dataset')
parser.add_argument('--filename', type=str, default='diplomacy_test.csv',
                   help='Input CSV file (default: diplomacy_test.csv)')
args = parser.parse_args()

df_test = pd.read_csv(args.filename)

print("[>] Preprocessing test data...")
df_test['clean_text'] = df_test['text'].apply(clean_text)

print("[>] Loading vectorizers...")
try:
    vectorizer_bow = joblib.load('vectorizer_bow.pkl')
    vectorizer_char = joblib.load('vectorizer_char.pkl')
except FileNotFoundError as e:
    raise FileNotFoundError("[X] Vectorizer files not found. Please run training script first.") from e

print("[>] Transforming test data...")
X_test_bow = vectorizer_bow.transform(df_test['clean_text'])
X_test_char = vectorizer_char.transform(df_test['clean_text'])

print("[>] Loading models...")
try:
    bow_model = joblib.load('logistic_bow_model.pkl')
    bow_selected_model = joblib.load('logistic_bow_selected_model.pkl')
    char_selected_model = joblib.load('logistic_char_selected_model.pkl')
except FileNotFoundError as e:
    raise FileNotFoundError("[X] Model files not found - Please run training script first!") from e

feature_selector = SelectKBest(chi2, k=1000)
X_test_bow_selected = feature_selector.fit_transform(X_test_bow, np.zeros(X_test_bow.shape[0]))

print("[>] Generating predictions...")
y_pred_bow = bow_model.predict(X_test_bow)
y_pred_bow_selected = bow_selected_model.predict(X_test_bow_selected)
y_pred_char = char_selected_model.predict(X_test_char)

print("\n[>] Loading neural network model...")
try:
    nn_model = NeuralNet(1000)
    nn_model.load_state_dict(torch.load('neural_net_model.pth'))
    nn_model.eval()
except FileNotFoundError as e:
    raise FileNotFoundError("Neural network model file not found. Please run training script first.") from e

print("[>] Generating neural network predictions...")
X_test_tensor = torch.tensor(X_test_bow_selected.toarray(), dtype=torch.float32)
with torch.no_grad():
    outputs = nn_model(X_test_tensor)
    _, y_pred_nn = torch.max(outputs, 1)
    y_pred_nn = y_pred_nn.numpy()

print("\n[!] Saving predictions...")

model_predictions = {
    'bow_unigram': y_pred_bow,
    'bow_selected': y_pred_bow_selected,
    'char_tfidf': y_pred_char,
    'neural_net': y_pred_nn
}

for model_name, preds in model_predictions.items():
    submission = pd.DataFrame({
        'id': df_test['id'],
        'intent': preds
    })
    filename = f'qkw3_{model_name}_predictions.csv'
    submission.to_csv(filename, index=False)
    print(f"[>] Saved predictions to {filename}")

print("\n[!] All predictions have been saved!")

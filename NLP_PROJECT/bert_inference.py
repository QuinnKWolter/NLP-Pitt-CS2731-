import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import time
from datetime import datetime
import os

def save_batch(data, batch_num, output_dir="output"):
    """Save a batch of predictions to a CSV file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.DataFrame(data)
    filename = f"{output_dir}/predictions_batch_{batch_num}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved batch {batch_num} to {filename}")

def predict_topic(text, tokenizer, model, label_encoder, device):
    """Predict topic for a single text entry"""
    try:
        # Convert to string if not already
        text = str(text) if text is not None else ""
        
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()

        # Decode the label
        decoded_label = label_encoder.inverse_transform([predicted_label])[0]
        major_topic, subtopic = decoded_label.split('-')
        return major_topic, subtopic
    
    except Exception as e:
        print(f"Error processing text: {text[:100]}...")  # Print first 100 chars of problematic text
        print(f"Error message: {str(e)}")
        return "ERROR", "ERROR"

# Start timing
start_time = time.time()
print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load the Twitter posts dataset
print("Loading Twitter posts data...")
twitter_df = pd.read_csv('twitter_posts.csv')
print(f"Twitter posts data loaded successfully. {len(twitter_df)} posts found.")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# Load the trained model and tokenizer
print("Loading model and tokenizer...")
model = BertForSequenceClassification.from_pretrained('bert_topic_classifier')
tokenizer = BertTokenizer.from_pretrained('bert_topic_classifier')
model.eval()
print("Model and tokenizer loaded.")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# Load the label encoder
print("Loading label encoder...")
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("Label encoder loaded.")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Prepare the output data
output_data = []
batch_size = 100000  # Save every 100,000 entries
batch_num = 1

# Iterate over each post and predict topics
print("Starting predictions...")
for idx, row in twitter_df.iterrows():
    if idx > 0 and idx % 1000 == 0:  # Progress update every 1000 entries
        elapsed_time = time.time() - start_time
        print(f"Processed {idx}/{len(twitter_df)} entries. Time elapsed: {elapsed_time:.2f} seconds")
    
    try:
        major_topic, subtopic = predict_topic(row['text'], tokenizer, model, label_encoder, device)
        
        output_data.append({
            'post_id': row['id'],
            'legislator_id': row['lid'],
            'major_topic': major_topic,
            'subtopic': subtopic,
            'text': row['text']
        })
        
        # Save batch if we've reached batch_size
        if len(output_data) >= batch_size:
            save_batch(output_data, batch_num)
            output_data = []  # Clear the list after saving
            batch_num += 1
            
    except Exception as e:
        print(f"Error processing row {idx}:")
        print(f"Row content: {row}")
        print(f"Error message: {str(e)}")
        continue

# Save any remaining data
if output_data:
    save_batch(output_data, batch_num)

# Combine all batches into final output
print("Combining all batches into final output...")
all_files = [f"output/predictions_batch_{i}.csv" for i in range(1, batch_num + 1)]
combined_df = pd.concat([pd.read_csv(f) for f in all_files])

# Save the final combined results
combined_df.to_csv('twitter_posts_with_topics.csv', index=False)
print("Results saved to twitter_posts_with_topics.csv")

end_time = time.time()
total_time = end_time - start_time
print(f"Processing completed. Total time: {total_time:.2f} seconds") 
import pandas as pd

# Load files
codebook_df = pd.read_csv('codebook.csv')
platform_rep_df = pd.read_csv('platform_rep.csv')
platform_dem_df = pd.read_csv('platform_dem.csv')

# Combine platform data
combined_platform_df = pd.concat([platform_rep_df, platform_dem_df], ignore_index=True)
combined_platform_df.rename(columns={'description': 'text'}, inplace=True)

# Trim '00' from 'Major Topic Code' for proper matching
codebook_df['majortopic_code'] = (codebook_df['Major Topic Code'] // 100).astype(int)
codebook_df['subtopic'] = codebook_df['Subtopic Code']

# Ensure majortopic_code is integer type
codebook_df['majortopic_code'] = codebook_df['majortopic_code'].astype(int)

# Prepare label mapping from codebook
label_mapping = codebook_df[['majortopic_code', 'Major Topic', 'subtopic', 'Subtopic']].drop_duplicates()
label_mapping.rename(columns={'Subtopic': 'subtopic_code'}, inplace=True)

# Merge platform data with codebook labels
combined_labeled_df = pd.merge(combined_platform_df, label_mapping, left_on=['majortopic', 'subtopic'], right_on=['majortopic_code', 'subtopic'], how='left')

# Replace missing labels with 'Unknown'
combined_labeled_df['majortopic'] = combined_labeled_df['Major Topic'].fillna('Unknown')
combined_labeled_df['subtopic_code'] = combined_labeled_df['subtopic_code'].fillna('Unknown')

# Select relevant columns with subtopic/subtopic_code swapped
training_data = combined_labeled_df[['majortopic', 'majortopic_code', 'subtopic_code', 'subtopic', 'text']]

# Save the combined training data to a CSV file
training_data.to_csv('combined_training_data.csv', index=False)

print('Combined training data saved to combined_training_data.csv')

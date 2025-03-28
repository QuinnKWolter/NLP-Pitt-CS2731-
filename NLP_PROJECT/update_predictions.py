import pandas as pd
import os
import glob

def load_codebook():
    """Load and prepare the codebook for lookups"""
    print("Loading codebook...")
    codebook = pd.read_csv('codebook.csv')
    
    # Create dictionaries for quick lookups
    major_topic_labels = {}
    subtopic_labels = {}
    
    for _, row in codebook.iterrows():
        # Major topic code in codebook is 100x larger than in predictions
        major_code = row['Major Topic Code'] // 100
        major_topic_labels[major_code] = row['Major Topic']
        
        subtopic_key = f"{major_code}-{row['Subtopic Code']}"
        subtopic_labels[subtopic_key] = row['Subtopic']
    
    print("Codebook prepared for lookups")
    return major_topic_labels, subtopic_labels

def update_batch_file(file_path, major_topic_labels, subtopic_labels):
    """Update a single batch file with topic labels"""
    print(f"\nProcessing {file_path}...")
    
    try:
        # Read the batch file
        df = pd.read_csv(file_path)
        
        # Convert major_topic to integer
        df['major_topic'] = df['major_topic'].astype(float).astype(int)
        
        # Add new columns
        df['major_topic_label'] = df['major_topic'].map(major_topic_labels)
        
        # Create subtopic keys and map to labels
        df['subtopic_key'] = df['major_topic'].astype(str) + '-' + df['subtopic'].astype(str)
        df['subtopic_label'] = df['subtopic_key'].map(subtopic_labels)
        
        # Drop the temporary key column
        df = df.drop('subtopic_key', axis=1)
        
        # Reorder columns
        columns = ['post_id', 'legislator_id', 'major_topic', 'major_topic_label', 
                  'subtopic', 'subtopic_label', 'text']
        df = df[columns]
        
        # Save updated file
        output_path = file_path.replace('.csv', '_updated.csv')
        df.to_csv(output_path, index=False)
        print(f"Successfully updated {file_path}")
        print(f"Saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def combine_updated_files():
    """Combine all updated batch files into a single file"""
    print("\nCombining updated files...")
    
    # Get all updated batch files
    updated_files = glob.glob('output/*_updated.csv')
    
    if not updated_files:
        print("No updated files found!")
        return
    
    try:
        # Read and combine all files
        dfs = []
        for file in updated_files:
            print(f"Reading {file}...")
            df = pd.read_csv(file)
            dfs.append(df)
        
        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save combined file
        output_path = 'predictions_combined.csv'
        combined_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully combined {len(updated_files)} files!")
        print(f"Combined file saved to: {output_path}")
        print(f"Total records: {len(combined_df)}")
        
    except Exception as e:
        print(f"Error combining files: {str(e)}")

def main():
    # Load codebook lookups
    major_topic_labels, subtopic_labels = load_codebook()
    
    # Get all batch files in the output directory
    batch_files = glob.glob('output/predictions_batch_*.csv')
    
    if not batch_files:
        print("No batch files found in the output directory!")
        return
    
    print(f"Found {len(batch_files)} batch files to process")
    
    # Process each batch file
    for file_path in batch_files:
        update_batch_file(file_path, major_topic_labels, subtopic_labels)
    
    print("\nAll batch files processed!")
    
    # Combine all updated files
    combine_updated_files()

if __name__ == "__main__":
    main() 
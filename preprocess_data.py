import pandas as pd
import os
import time

# Define paths
data_path = os.path.join('data', 'Reviews.csv')
output_path = os.path.join('data', 'preprocessed_reviews.csv')

# Check if the input file exists
if not os.path.exists(data_path):
    print(f"Error: Dataset file not found at {data_path}")
else:
    print(f"Loading dataset from: {data_path}")
    start_time = time.time()
    # Load only necessary columns to save memory
    df = pd.read_csv(data_path, usecols=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text'])
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")
    print(f"Initial shape: {df.shape}")

    # --- Data Cleaning ---

    # 1. Handle Missing Values
    print("\nChecking for missing values...")
    print(df.isnull().sum())
    # Drop rows where 'Text' or 'Score' is missing (if any)
    df.dropna(subset=['Text', 'Score'], inplace=True)
    print(f"Shape after dropping rows with missing Text/Score: {df.shape}")

    # 2. Remove Duplicates
    # Reviews can be duplicated if a user posts the same review for different products
    # or if there are actual data entry errors.
    # A common approach is to remove entries where UserId, ProfileName, Time, and Text are identical.
    print("\nRemoving duplicate entries...")
    initial_shape = df.shape
    df.drop_duplicates(subset=['UserId', 'ProfileName', 'Time', 'Text'], keep='first', inplace=True)
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicate entries.")
    print(f"Shape after removing duplicates: {df.shape}")

    # --- Preprocessing for Sentiment Analysis ---

    # 3. Filter out neutral reviews (Score = 3)
    print("\nFiltering out neutral reviews (Score = 3)...")
    df_filtered = df[df['Score'] != 3].copy() # Use .copy() to avoid SettingWithCopyWarning
    print(f"Shape after removing neutral reviews: {df_filtered.shape}")

    # 4. Create Sentiment Label
    # Score > 3 is Positive (1), Score < 3 is Negative (0)
    print("Creating binary sentiment labels...")
    df_filtered['Sentiment'] = df_filtered['Score'].apply(lambda x: 1 if x > 3 else 0)

    # 5. Select Relevant Columns
    print("Selecting 'Text' and 'Sentiment' columns...")
    df_final = df_filtered[['Text', 'Sentiment']].copy()

    # --- Final Checks ---
    print("\nPreprocessing complete.")
    print(f"Final dataset shape: {df_final.shape}")
    print("\nValue counts for Sentiment:")
    print(df_final['Sentiment'].value_counts())

    # Optional: Save the preprocessed data
    print(f"\nSaving preprocessed data to: {output_path}")
    try:
        df_final.to_csv(output_path, index=False)
        print("Preprocessed data saved successfully.")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")

    end_time = time.time()
    print(f"\nTotal preprocessing time: {end_time - start_time:.2f} seconds.")
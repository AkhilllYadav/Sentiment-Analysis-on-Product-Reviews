import re
import time
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # Import Lemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# --- Configuration ---
preprocessed_file = os.path.join('data', 'preprocessed_reviews.csv')
output_dir = 'processed_data'
vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.joblib')
X_train_path = os.path.join(output_dir, 'X_train.joblib')
X_test_path = os.path.join(output_dir, 'X_test.joblib')
y_train_path = os.path.join(output_dir, 'y_train.joblib')
y_test_path = os.path.join(output_dir, 'y_test.joblib')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Load Preprocessed Data ---
print(f"Loading preprocessed data from: {preprocessed_file}")
start_time = time.time()
try:
    df = pd.read_csv(preprocessed_file)
    # Drop rows where CleanedText might be NaN after preprocessing (if any)
    df.dropna(subset=['CleanedText'], inplace=True)
    print(f"Data loaded successfully in {time.time() - start_time:.2f} seconds.")
except FileNotFoundError:
    print(f"Error: File not found at {preprocessed_file}")
    print("Please ensure 'preprocess_data.py' ran successfully.")
    exit()
except Exception as e:
    print(f"An error occurred loading data: {e}")
    exit()


# --- Text Cleaning Function ---
print("Setting up text cleaning utilities...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() # Initialize Lemmatizer

def clean_text(text):
    """Cleans text data by removing HTML tags, non-alphabetic characters,
    converting to lowercase, removing stop words, and lemmatizing."""
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters and numbers, keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stop words, then lemmatize
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words] # Lemmatize here
    return ' '.join(cleaned_words)

# --- Apply Text Cleaning ---
print("Applying text cleaning to 'CleanedText' column...")
cleaning_start_time = time.time()
# Ensure the column exists before applying
if 'CleanedText' in df.columns:
    df['CleanedText'] = df['CleanedText'].apply(clean_text)
    print(f"Text cleaning completed in {time.time() - cleaning_start_time:.2f} seconds.")
    # Display first few cleaned texts
    print("\nSample cleaned texts:")
    print(df['CleanedText'].head())
else:
    print("Error: 'CleanedText' column not found in the preprocessed data.")
    exit()


# --- Train-Test Split ---
print("\nSplitting data into training and testing sets...")
split_start_time = time.time()
X = df['CleanedText']
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split completed in {time.time() - split_start_time:.2f} seconds.")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


# --- TF-IDF Vectorization ---
print("\nPerforming TF-IDF vectorization...")
vectorizer_start_time = time.time()
# Limit features to potentially improve performance and reduce memory
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2)) # Example: Use top 10k features, uni+bigrams

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"TF-IDF vectorization completed in {time.time() - vectorizer_start_time:.2f} seconds.")
print(f"Shape of TF-IDF matrix (Train): {X_train_tfidf.shape}")
print(f"Shape of TF-IDF matrix (Test): {X_test_tfidf.shape}")

# --- Save Processed Data and Vectorizer ---
print(f"\nSaving processed data and vectorizer to: {output_dir}")
save_start_time = time.time()
try:
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    joblib.dump(X_train_tfidf, X_train_path)
    joblib.dump(X_test_tfidf, X_test_path)
    joblib.dump(y_train, y_train_path)
    joblib.dump(y_test, y_test_path)
    print(f"Files saved successfully in {time.time() - save_start_time:.2f} seconds.")
except Exception as e:
    print(f"Error saving files: {e}")

end_time = time.time()
print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")
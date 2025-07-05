import pandas as pd
import os

# Define the path to the dataset
# Adjust the path if your script is not in the root directory
data_path = os.path.join('data', 'Reviews.csv')

# Check if the file exists
if not os.path.exists(data_path):
    print(f"Error: Dataset file not found at {data_path}")
else:
    try:
        # Load the dataset
        print(f"Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)

        # Display the first 5 rows
        print("\nFirst 5 rows of the dataset:")
        print(df.head())

        # Display basic information (column names, non-null counts, data types)
        print("\nDataset Info:")
        df.info()

        # Display descriptive statistics
        print("\nDescriptive Statistics:")
        print(df.describe(include='all')) # Include 'all' to get stats for non-numeric columns too

        # Display the shape of the dataset (rows, columns)
        print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    except Exception as e:
        print(f"An error occurred while loading or processing the data: {e}")
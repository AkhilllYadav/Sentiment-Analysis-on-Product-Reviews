import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
preprocessed_file = os.path.join('data', 'preprocessed_reviews.csv')
output_dir = 'models'
output_path = os.path.join(output_dir, 'sentiment_distribution_pie.png')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
print(f"Loading preprocessed data from: {preprocessed_file}")
try:
    df = pd.read_csv(preprocessed_file)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {preprocessed_file}")
    print("Please ensure 'preprocess_data.py' ran successfully and the file exists.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- Calculate Sentiment Counts ---
sentiment_counts = df['Sentiment'].value_counts()
labels = ['Positive (1)', 'Negative (0)']
sizes = [sentiment_counts.get(1, 0), sentiment_counts.get(0, 0)] # Use .get() for safety if one class is missing
colors = ['#66b3ff', '#ff9999'] # Light blue for positive, light red for negative
explode = (0.05, 0) # Slightly explode the 'Positive' slice

# --- Generate Pie Chart ---
print("Generating pie chart...")
plt.figure(figsize=(7, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Distribution of Sentiments in Preprocessed Data')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

# --- Save Chart ---
try:
    plt.savefig(output_path)
    print(f"Pie chart saved successfully to: {output_path}")
except Exception as e:
    print(f"An error occurred while saving the chart: {e}")

# plt.show() # Uncomment to display the plot directly
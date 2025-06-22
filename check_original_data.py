import pandas as pd

# Path to your original training data CSV file
ORIGINAL_TRAINING_DATA_PATH = 'dataset/new_data_urls.csv' # Adjust path if needed

# Load the CSV file
df_original_data = pd.read_csv(ORIGINAL_TRAINING_DATA_PATH)

# Display the first few rows and information about the DataFrame
print("First 5 rows of new_data_urls.csv:")
print(df_original_data.head())
print("\nInformation about new_data_urls.csv:")
print(df_original_data.info())
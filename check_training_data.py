import pandas as pd
import os

# Path to your final combined training data
TRAINING_DATA_PATH = 'final_combined_training_data.csv'

# URLs to check
urls_to_check = [
    "https://www.nic.in/",
    "http://login-verify-paypal.com.account-update.xyz/login",
    "https://www.google.com/"
]

print(f"📁 Loading training data from {TRAINING_DATA_PATH}...")
if os.path.exists(TRAINING_DATA_PATH):
    df_training = pd.read_csv(TRAINING_DATA_PATH)
    print(f"✅ Loaded {len(df_training)} URLs from the training dataset.")
else:
    print(f"Error: Training data file not found at '{TRAINING_DATA_PATH}'.")
    print("Please ensure 'final_combined_training_data.csv' exists in your project folder.")
    exit()

print("\n--- Checking URLs in Training Data ---")
for url in urls_to_check:
    # Normalize URL for consistent lookup (e.g., remove trailing slashes if not significant)
    # Be careful with normalization, as sometimes a trailing slash changes meaning.
    # For exact match, no special normalization is done here.
    
    found_row = df_training[df_training['url'] == url]
    
    if not found_row.empty:
        label = found_row['label'].iloc[0]
        status = "SAFE" if label == 0 else "PHISHING"
        print(f"'{url}' found in training data. Label: {label} ({status})")
    else:
        print(f"'{url}' NOT found in training data.")

print("\n--- Check Complete ---")
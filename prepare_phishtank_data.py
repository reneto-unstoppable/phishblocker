import pandas as pd
import os

# Path to the downloaded gzipped CSV file from PhishTank
PHISHTANK_GZ_PATH = 'dataset/online-valid.csv.gz'

# Output path for your new phishing URLs file
OUTPUT_PHISHING_URLS_PATH = 'dataset/new_phishtank_urls_for_training.csv'

print(f"📁 Loading PhishTank data from {PHISHTANK_GZ_PATH}...")
try:
    # Read the gzipped CSV directly. pandas can handle .gz files.
    # We only need 'url' and implicitly assign 'label' = 1
    df_phishtank = pd.read_csv(PHISHTANK_GZ_PATH, usecols=['url'])

    # Add a 'label' column with value 1 (for phishing) to all these URLs
    df_phishtank['label'] = 1

    print(f"✅ Loaded {len(df_phishtank)} URLs from PhishTank.")
    print("First 5 URLs from PhishTank data:")
    print(df_phishtank.head())

    # Save the prepared data to a new CSV file
    df_phishtank.to_csv(OUTPUT_PHISHING_URLS_PATH, index=False)
    print(f"\n✅ Prepared PhishTank URLs saved to '{OUTPUT_PHISHING_URLS_PATH}'.")

except FileNotFoundError:
    print(f"Error: The file '{PHISHTANK_GZ_PATH}' was not found.")
    print("Please ensure you have downloaded 'online-valid.csv.gz' and placed it in your 'dataset' folder.")
except Exception as e:
    print(f"An error occurred: {e}")
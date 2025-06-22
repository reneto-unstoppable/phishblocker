import pandas as pd
import sys
import os

# Add the parent directory to the path to import feature_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Adjust the path above based on your exact file structure
# If feature_utils.py is in the same directory as this script, you might just need:
from feature_utils import extract_features # Assuming feature_utils is in the same directory

# Or, if feature_utils.py is in the 'phishblocker' directory and this script is also there,
# the original import should work if you run it from the phishblocker root.
# Let's assume you run this from the 'phishblocker' root.
# from feature_utils import extract_features


# List of problematic URLs to inspect
urls_to_inspect = [
    "https://www.microsoft.com/",
    "https://en.wikipedia.org/wiki/Main_Page",
    "http://my-banking.ru/login.php?user=secure",
    "https://www.amazon.com.security-alert.xyz/verify"
]

print("--- Extracting Features for Problematic URLs ---")

for url in urls_to_inspect:
    print(f"\nURL: {url}")
    try:
        features_df, _ = extract_features(url)
        if not features_df.empty:
            # Transpose the DataFrame to make it easier to read features vertically
            features_df_transposed = features_df.T
            features_df_transposed.columns = ['Value']
            print(features_df_transposed)
        else:
            print("Failed to extract features for this URL.")
    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")

print("\n--- Feature Inspection Complete ---")
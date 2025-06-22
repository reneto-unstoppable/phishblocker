import pandas as pd
import os

# Define the path to your CSV file
csv_file_path = 'final_combined_training_data.csv' # Make sure this matches your file name

if not os.path.exists(csv_file_path):
    print(f"Error: '{csv_file_path}' not found in the current directory.")
    print("Please make sure the script is in the same directory as your CSV, or provide the full path.")
else:
    try:
        df = pd.read_csv(csv_file_path)

        print(f"--- Analysis of {csv_file_path} ---")

        # 1. Total number of rows
        total_rows = len(df)
        print(f"Total number of URLs: {total_rows}")

        # 2. Count of safe (0) and phishing (1) URLs
        if 'label' in df.columns: # Assuming your label column is named 'label'
            label_counts = df['label'].value_counts()
            print("\nLabel Distribution:")
            if 0 in label_counts:
                print(f"  Safe (0): {label_counts[0]} URLs")
            else:
                print("  Safe (0): 0 URLs (No safe URLs found in labels)")

            if 1 in label_counts:
                print(f"  Phishing (1): {label_counts[1]} URLs")
            else:
                print("  Phishing (1): 0 URLs (No phishing URLs found in labels)")

            # 3. Print a few example URLs for each category
            print("\n--- Example Safe URLs (label=0) ---")
            safe_urls = df[df['label'] == 0]['url'].head(5) # Get first 5 safe URLs
            if not safe_urls.empty:
                for url in safe_urls:
                    print(f"- {url}")
            else:
                print("No safe URLs found in the dataset.")

            print("\n--- Example Phishing URLs (label=1) ---")
            phishing_urls = df[df['label'] == 1]['url'].head(5) # Get first 5 phishing URLs
            if not phishing_urls.empty:
                for url in phishing_urls:
                    print(f"- {url}")
            else:
                print("No phishing URLs found in the dataset.")

        else:
            print(f"Error: 'label' column not found in '{csv_file_path}'. Please check your column names.")

    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
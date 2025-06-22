import pandas as pd
import os

# --- Configuration ---
# Path to your final, comprehensive combined training data file
EXISTING_FINAL_DATA_PATH = 'final_combined_training_data.csv'

# Path to the new manual corrections file
MANUAL_CORRECTIONS_PATH = 'dataset/manual_corrections.csv' # ADDED THIS LINE

# Name for the ultimate, corrected training data file
# We will overwrite the old final_combined_training_data.csv
OUTPUT_TRAINING_DATA_PATH = 'final_combined_training_data.csv'

# --- Step 1: Load Existing Final Training Data ---
print(f"📁 Loading existing comprehensive training data from {EXISTING_FINAL_DATA_PATH}...")
if os.path.exists(EXISTING_FINAL_DATA_PATH):
    df_existing_final = pd.read_csv(EXISTING_FINAL_DATA_PATH)
    print(f"Loaded existing final training data: {len(df_existing_final)} rows.")
else:
    print(f"Error: '{EXISTING_FINAL_DATA_PATH}' not found. Please ensure it was created previously.")
    exit()

# --- Step 2: Load Manual Corrections ---
print(f"\n📁 Loading manual corrections from '{MANUAL_CORRECTIONS_PATH}'...")
if os.path.exists(MANUAL_CORRECTIONS_PATH):
    df_manual_corrections = pd.read_csv(MANUAL_CORRECTIONS_PATH)
    print(f"Loaded manual corrections: {len(df_manual_corrections)} rows.")
else:
    print(f"Error: '{MANUAL_CORRECTIONS_PATH}' not found. Please ensure you created it in the 'dataset' folder.")
    exit()

# --- Step 3: Combine all URLs and Deduplicate, ensuring corrections take precedence ---
# Concatenate existing final data with the manual corrections
# Manual corrections should conceptually overwrite or take precedence if duplicates exist
df_ultimate_combined = pd.concat([df_existing_final, df_manual_corrections], ignore_index=True)

# Important: When deduplicating, we want the LAST occurrence (our manual correction) to win.
# So, we sort such that manual corrections appear later, then drop duplicates keeping the last.
df_ultimate_combined.sort_values(by='url', ascending=True, kind='mergesort', inplace=True) # Stable sort
df_ultimate_combined.drop_duplicates(subset=['url'], keep='last', inplace=True) # Keep the manual correction

initial_rows_before_final_dedupe = len(df_ultimate_combined)
duplicates_removed = initial_rows_before_final_dedupe - len(df_ultimate_combined)

print(f"\nCombined data has {initial_rows_before_final_dedupe} rows before final deduplication (including corrections).")
print(f"Removed {duplicates_removed} duplicate URLs, prioritizing manual corrections.")
print(f"Ultimate comprehensive training data has {len(df_ultimate_combined)} unique URLs.")

# --- Step 4: Save the Ultimate Training Data ---
df_ultimate_combined.to_csv(OUTPUT_TRAINING_DATA_PATH, index=False)
print(f"\n✅ Ultimate comprehensive training data saved to '{OUTPUT_TRAINING_DATA_PATH}'.")

print("\nFirst 5 rows of the ultimate combined training data:")
print(df_ultimate_combined.head())
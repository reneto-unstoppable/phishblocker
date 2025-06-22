import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
from tqdm import tqdm
import os
import sys
import numpy as np
from feature_utils import extract_features # Assuming feature_utils.py is in the same directory

# --- Configuration ---
DATASET_PATH = 'expanded_training_data.csv' # Corrected path based on your input
MODEL_OUTPUT_PATH = 'phishing_model.pkl'
SCALER_OUTPUT_PATH = 'scaler.pkl'

# --- 📁 Load Dataset ---
print("📁 Loading expanded dataset...")
try:
    df_expanded = pd.read_csv(DATASET_PATH)
    # Ensure URL column is string type
    df_expanded['url'] = df_expanded['url'].astype(str)
    # Drop rows where 'url' might be empty or problematic
    df_expanded = df_expanded[df_expanded['url'].str.strip() != ''].copy()

    if df_expanded.empty:
        raise ValueError(f"Dataset '{DATASET_PATH}' is empty or contains no valid URLs after cleaning.")

except FileNotFoundError:
    print(f"Error: Dataset not found at {DATASET_PATH}. Please make sure the file exists.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# --- DEBUG: Initial Label Distribution ---
print("📊 Initial label distribution in the expanded dataset (before inversion):")
print(df_expanded['label'].value_counts().reset_index())
print(df_expanded['label'].value_counts(normalize=True))

# --- !!! CRITICAL FIX: INVERT LABELS !!! ---
# If 0 in your CSV means Phishing and 1 means Safe, we need to flip them
# so that for our model and interpretation: 0 = Safe, 1 = Phishing.
# This operation assumes the original labels are strictly 0 and 1.
print("\n🔄 Inverting labels (0s become 1s, 1s become 0s) to align with convention (0=Safe, 1=Phishing)...")

print("📊 Label distribution AFTER inversion:")
print(df_expanded['label'].value_counts().reset_index())
print(df_expanded['label'].value_counts(normalize=True))
# --- END CRITICAL FIX ---


# --- 🔄 Feature Extraction ---
print("\n🔄 Extracting features from URLs...")
features_list = []
labels_list = []
failed_extractions = 0

# Use tqdm for a progress bar
for index, row in tqdm(df_expanded.iterrows(), total=len(df_expanded), desc="Extracting features from URLs"):
    url = row['url']
    label = row['label']
    
    features_df, extraction_error = extract_features(url)

    if not features_df.empty:
        features_list.append(features_df.iloc[0])
        labels_list.append(label)
    else:
        failed_extractions += 1

if not features_list:
    print("\nError: No features were successfully extracted. Please check feature_utils.py for issues.")
    print(f"⚠️ Total failed extractions: {failed_extractions}")
    print(f"✅ Successfully extracted features from 0 URLs")
    sys.exit(1)

X = pd.DataFrame(features_list)
y = pd.Series(labels_list)

print(f"\n⚠️ Total failed extractions: {failed_extractions}")
print(f"✅ Successfully extracted features from {len(X)} URLs")

# Handle potential NaN or infinite values in features after extraction
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)


# --- splitting data into training and test sets ---
print("\nsplitting data into training and test sets (80/20 split)...")
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError as e:
    print(f"Error splitting data: {e}")
    print("This often happens if X or y is empty after feature extraction.")
    sys.exit(1)

# --- ⚖️ Balancing training dataset using SMOTE ---
print("\n⚖️ Balancing training dataset using SMOTE... (will balance based on inverted labels)")
print(f"Original training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Original training label distribution: {Counter(y_train)}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled training set shape: X_train_resampled {X_train_resampled.shape}, y_train_resampled {y_train_resampled.shape}")
print(f"Resampled training label distribution: {Counter(y_train_resampled)}")

# --- DEBUGGING LABELS ---
print(f"DEBUG: y_train_resampled unique values and their counts: {Counter(y_train_resampled)}")
# --- END DEBUGGING LABELS ---

# --- 📏 Scale Features ---
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# --- 🧠 Train Random Forest Model ---
print("\n🧠 Training Random Forest model on resampled data...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train_resampled)

# --- ✅ Save Model and Scaler ---
joblib.dump(model, MODEL_OUTPUT_PATH)
joblib.dump(scaler, SCALER_OUTPUT_PATH)
print(f"\n✅ Model trained and saved as {MODEL_OUTPUT_PATH}")
print(f"✅ Scaler saved as {SCALER_OUTPUT_PATH}")

# --- DEBUGGING MODEL CLASSES ---
print(f"DEBUG: Model classes: {model.classes_}")
# --- END DEBUGGING MODEL CLASSES ---

# --- Internal Test Set Evaluation ---
print("\n--- Internal Test Set Evaluation ---")
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print("--- End Internal Test Set Evaluation ---")
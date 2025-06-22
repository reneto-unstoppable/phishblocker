import pandas as pd
import joblib
from sklearn.metrics import classification_report
from feature_utils import extract_features
import sys
import numpy as np # Import numpy

# Load model
model = joblib.load("phishing_model.pkl")

# Load external CSV
if len(sys.argv) != 2:
    print("Usage: python evaluate_model.py <csv_file>")
    sys.exit()

input_csv = sys.argv[1]
df = pd.read_csv(input_csv)

# Extract & flatten features properly
X = pd.concat(df['url'].apply(lambda url: extract_features(url)[0]).to_list(), axis=0, ignore_index=True)
y_true = df['label'] # y_true correctly assigned here

# Predict probabilities instead of direct classes
# model.predict_proba returns probabilities for [class 0, class 1]
y_pred_proba = model.predict_proba(X)

# Define a custom threshold for predicting class 1 (Phishing)
# Let's start by trying a lower threshold, e.g., 0.3.
# This means if the probability of being phishing is > 0.3, we'll classify it as phishing.
PHISHING_THRESHOLD = 0.01 # Let's try to get more phishing URLs! # <--- YOU CAN EXPERIMENT WITH THIS VALUE (e.g., 0.2, 0.4)

# Apply the custom threshold to get predictions
# y_pred will be 1 if P(class=1) > PHISHING_THRESHOLD, else 0
y_pred = (y_pred_proba[:, 1] > PHISHING_THRESHOLD).astype(int)

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

# Feature Importance (this part remains the same)
feature_names = X.columns.tolist()
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n🚨 Feature Importances:")
print(importance_df)
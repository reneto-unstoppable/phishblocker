import pandas as pd

# Load dataset
df = pd.read_csv("static/dataset/malicious_phish.csv")

# Basic info
print("✅ Dataset loaded successfully!\n")
print("🔹 First 5 rows:")
print(df.head())
print("\n🔹 Dataset info:")
print(df.info())
print("\n🔹 Value counts (label distribution):")
print(df['type'].value_counts())

import pandas as pd

# 📂 Load the new dataset
df = pd.read_csv("dataset/new_data_urls.csv")

# 🔍 Check structure
print("📄 Column names:", df.columns.tolist())
print("\n🧪 First 5 rows:")
print(df.head())

# 📊 Label distribution (status instead of label)
print("\n📊 Label distribution:")
print(df['status'].value_counts())

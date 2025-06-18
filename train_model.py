import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from tqdm import tqdm

# 💡 Feature extraction logic
def extract_features(url):
    url_length = len(url)
    num_digits = sum(c.isdigit() for c in url)
    num_special_chars = sum(not c.isalnum() for c in url)
    num_subdomains = url.count('.') - 1
    count_at = url.count('@')
    count_com = url.count('.com')
    count_hyphen = url.count('-')
    count_net = url.count('.net')
    count_www = url.count('www')
    has_https = int(url.startswith("https"))

    features = {
        "url_length": url_length,
        "num_digits": num_digits,
        "num_special_chars": num_special_chars,
        "num_subdomains": num_subdomains,
        "count_at": count_at,
        "count_com": count_com,
        "count_hyphen": count_hyphen,
        "count_net": count_net,
        "count_www": count_www,
        "has_https": has_https,
    }

    return pd.DataFrame([features])

# 📁 Load dataset
print("📁 Loading dataset...")
df = pd.read_csv("dataset/new_data_urls.csv")

# 🏷️ Use 'status' column as label (0: legitimate, 1: phishing)
print("🔧 Mapping labels...")
df = df[df["status"].isin([0, 1])]  # Filter only known status values
df["label"] = df["status"]

# 🔍 Show label balance
print("\n📊 Label counts:\n", df["label"].value_counts())

# ✅ Extract features from URLs
print("🔄 Extracting features from URLs...")
features = [extract_features(url).iloc[0] for url in tqdm(df["url"])]
X = pd.DataFrame(features)
y = df["label"]

# 🧠 Train Random Forest model
print("🧠 Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
model.fit(X, y)

# 💾 Save model
joblib.dump(model, "phishing_model.pkl")
print("✅ Model trained and saved as phishing_model.pkl")

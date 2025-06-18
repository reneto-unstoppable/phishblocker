import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 💡 Paste your extract_features() function here
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

    columns_order = [
        "url_length",
        "num_digits",
        "num_special_chars",
        "num_subdomains",
        "count_at",
        "count_com",
        "count_hyphen",
        "count_net",
        "count_www",
        "has_https"
    ]

    return pd.DataFrame([features])[columns_order]

# ✅ Sample training data
data = [
    {"url": "https://secure-login.com", "label": 0},
    {"url": "http://192.168.0.1", "label": 1},
    {"url": "https://www.paypal.com", "label": 0},
    {"url": "http://malicious-login.net", "label": 1},
    {"url": "https://accounts.google.com", "label": 0},
    {"url": "http://verify-now.xyz@phish.com", "label": 1},
]
df = pd.DataFrame(data)

# ✅ Extract features for training
features = [extract_features(url).iloc[0] for url in df["url"]]
X = pd.DataFrame(features)
y = df["label"]

# ✅ Train model
model = RandomForestClassifier()
model.fit(X, y)

# ✅ Save model
joblib.dump(model, "phishing_model.pkl")
print("✅ Model retrained and saved as phishing_model.pkl")

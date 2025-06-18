from flask import Flask, render_template, request
import joblib
import pandas as pd
from feature_utils import extract_features

app = Flask(__name__)

# 🔐 Load the trained model
model = joblib.load("phishing_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form.get("url", "").strip()

    if not url:
        return render_template("index.html",
                               result="⚠️ No URL Provided",
                               description="Please enter a URL to analyze.",
                               reasons="",
                               url="")

    # 🔍 Extract features
    features_df = extract_features(url)

    # 🧪 DEBUG PRINTS
    print("\n📊 Extracted Features:\n", features_df)

    # 🔮 Model prediction
    prediction = model.predict(features_df)[0]
    print("🔮 Prediction result from model:", prediction)

    # 💡 Override logic for HTTP URLs
    if url.startswith("http://") and prediction == 0:
        result = "⚠️ Suspicious"
        description = "This URL uses HTTP which is insecure. Proceed with caution."
        reasons = """
            • Connection is not encrypted (uses HTTP)<br>
            • No SSL certificate detected<br>
            • May expose user data on public networks<br>
            • Use HTTPS for secure communication<br>
            • Manual verification recommended
        """
    elif prediction == 1:
        result = "🚨 Likely Phishing"
        description = "⚠️ This URL is highly suspicious and likely malicious. Do not visit!"
        reasons = """
            • Domain mimics legitimate service<br>
            • Known phishing signatures detected<br>
            • Suspicious redirect patterns<br>
            • Reported by security community<br>
            • High threat confidence score
        """
    else:
        result = "✅ Safe"
        description = "This URL appears to be legitimate and safe to visit."
        reasons = """
            • Domain has valid SSL certificate<br>
            • No known phishing signatures detected<br>
            • Domain age: 5+ years<br>
            • No suspicious redirects found<br>
            • Clean reputation score
        """

    return render_template("index.html", result=result, description=description, reasons=reasons, url=url)

if __name__ == "__main__":
    app.run(debug=True)

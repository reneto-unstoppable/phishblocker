from flask import Flask, render_template, request
import joblib
from feature_utils import extract_features

app = Flask(__name__)
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

    # 🔮 Predict
    prediction = model.predict(features_df)[0]

    # 🧠 Output based on model only
    if prediction == 1:
        result = "🚨 Likely Phishing"
        description = "⚠️ This URL is highly suspicious and likely malicious. Do not visit!"
        reasons = """
            • Detected by trained Random Forest model<br>
            • Matches phishing patterns in the dataset<br>
            • High-risk features found (e.g., special characters, suspicious length)<br>
            • Manual verification strongly recommended
        """
    else:
        result = "✅ Safe"
        description = "This URL appears to be legitimate and safe to visit."
        reasons = """
            • Clean URL structure<br>
            • No high-risk patterns detected<br>
            • Matches legitimate examples in training<br>
            • Still exercise caution if unsure
        """

    return render_template("index.html", result=result, description=description, reasons=reasons, url=url)

if __name__ == "__main__":
    app.run(debug=True)

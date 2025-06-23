from flask import Flask, render_template, request
import joblib
import os
import gdown
from feature_utils import extract_features
import numpy as np
import requests
import validators
from requests.exceptions import ConnectionError, Timeout, RequestException
import traceback
from urllib.parse import urlparse

app = Flask(__name__)

# 📦 Model info
MODEL_PATH = "phishing_model.pkl"
SCALER_PATH = "scaler.pkl"
MODEL_ID = "1G_wI_-aYjT9bvGYW7WweE7r8raaCSCux"  # Google Drive ID for model (UPDATE THIS WITH NEW ID AFTER RETRAINING)
SCALER_ID = "1LOVJCAdwsVrpRzhNvtjJVOwkP9wHQVj5"  # Google Drive ID for scaler

# ✨ Define the PHISHING_THRESHOLD here
PHISHING_THRESHOLD = 0.6 # Increased threshold to reduce false positives

# Global session for connection pooling
session = requests.Session()

def unshorten_url(url):
    """
    Attempts to unshorten a URL using requests.Session for better connection management.
    Returns the unshortened URL or the original URL if unshortening fails
    or if it's not a recognized shortened URL.
    """
    if not isinstance(url, str) or not url.strip():
        return ""

    if not validators.url(url):
        print(f"DEBUG: unshorten_url - Invalid URL format: '{url}'")
        return url

    known_shorteners_domains = [
        'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'shorte.st',
        'cutt.ly', 'is.gd', 'rebrand.ly', 'clck.ru', 'rb.gy'
    ]

    parsed_input_url = urlparse(url)
    input_domain = parsed_input_url.netloc.lower()

    is_original_from_shortener = any(shortener in input_domain for shortener in known_shorteners_domains)

    # Only attempt unshortening if it looks like a shortened URL
    if not is_original_from_shortener:
        print(f"DEBUG: unshorten_url - Not a known shortened URL: '{url}'. Returning original.")
        return url

    print(f"DEBUG: unshorten_url - Attempting to resolve: {url}")
    try:
        # Use the global session
        response = session.get(url, allow_redirects=True, timeout=5)
        final_url_from_response = response.url

        print(f"DEBUG: unshorten_url - Response status code: {response.status_code}")
        print(f"DEBUG: unshorten_url - Redirect history: {response.history}")
        print(f"DEBUG: unshorten_url - Final URL from requests: {final_url_from_response}")

        # Check if the final URL is different from the original and is valid
        if final_url_from_response != url and validators.url(final_url_from_response):
            print(f"DEBUG: unshorten_url - Successfully resolved to: {final_url_from_response}")
            return final_url_from_response
        else:
            # If no effective redirect occurred, or final URL is invalid, return the original
            print(f"DEBUG: unshorten_url - No effective unshortening for '{url}'. Returning original.")
            return url

    except (ConnectionError, Timeout, RequestException) as e:
        print(f"ERROR: unshorten_url - Network/Request error for '{url}': {e}")
        return url
    except Exception as e:
        print(f"ERROR: unshorten_url - Unexpected error for '{url}': {e}")
        traceback.print_exc() # Print full traceback for unexpected errors
        return url


# --- Download model and scaler from Google Drive if not already downloaded ---
print("Attempting to load model and scaler...")
try:
    if not os.path.exists(MODEL_PATH) and MODEL_ID:
        print(f"⬇️ Model not found locally. Downloading '{MODEL_PATH}' from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
    else:
        print(f"✅ Found local model '{MODEL_PATH}'.")

    if not os.path.exists(SCALER_PATH) and SCALER_ID:
        print(f"⬇️ Scaler not found locally. Downloading '{SCALER_PATH}' from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={SCALER_ID}", SCALER_PATH, quiet=False)
    else:
        print(f"✅ Found local scaler '{SCALER_PATH}'.")

except Exception as e:
    print(f"Error during model/scaler download/check: {e}")
    traceback.print_exc()
    exit()

# --- ✅ Load trained model and scaler ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully.")
except FileNotFoundError:
    print(
        f"Error: Model or scaler file not found. Make sure '{MODEL_PATH}' and '{SCALER_PATH}' exist in the same directory as app.py.")
    exit()
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    traceback.print_exc()
    exit()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html",
                           result=None,
                           description=None,
                           reasons=None,
                           original_input_url="",
                           final_display_url="")


@app.route("/analyze", methods=["POST"])
def analyze():
    user_input_url = request.form.get("url", "").strip()

    if not user_input_url:
        return render_template("index.html",
                               result="⚠️ No URL Provided",
                               description="Please enter a URL to analyze.",
                               reasons="",
                               original_input_url="",
                               final_display_url="")

    final_display_url = unshorten_url(user_input_url)
    actual_url_for_analysis = final_display_url

    TRUSTED_DOMAINS = [
        "www.nasa.gov", "nasa.gov",
        "www.google.com", "google.com",
        "www.microsoft.com", "microsoft.com",
        "www.apple.com", "apple.com",
        "www.amazon.com", "amazon.com",
        "www.stanford.edu", "stanford.edu",
        "www.harvard.edu", "harvard.edu",
        "www.usa.gov", "usa.gov",
        "www.nist.gov", "nist.gov",
        "www.who.int", "who.int",
        "www.wikipedia.org", "wikipedia.org",
        "www.github.com", "github.com",
        "www.linkedin.com", "linkedin.com",
        "www.openai.com", "openai.com",
        "www.stackoverflow.com", "stackoverflow.com",
        "chatgpt.com",
        "www.facebook.com", "facebook.com",
        "www.twitter.com", "twitter.com", "x.com",
        "www.youtube.com", "youtube.com", # Added youtube.com to whitelist
        "youtu.be", # Added youtu.be to whitelist for YouTube short links
        "www.reddit.com", "reddit.com",
        "www.instagram.com", "instagram.com",
        "www.netflix.com", "netflix.com",
        "www.bankofamerica.com", "bankofamerica.com",
        "www.chase.com", "chase.com",
        "www.wellsfargo.com", "wellsfargo.com",
        "www.paypal.com", "paypal.com",
        "www.ebay.com", "ebay.com",
        "www.spotify.com", "spotify.com",
        "phishblocker.onrender.com"
        "www.flipkart.com", "flipkart.com"
    ]

    try:
        parsed_url_for_analysis = urlparse(actual_url_for_analysis)
        domain_to_check = parsed_url_for_analysis.netloc.lower()

        if domain_to_check.startswith("www."):
            domain_to_check = domain_to_check[4:]

        if domain_to_check in TRUSTED_DOMAINS:
            return render_template("index.html",
                                   result="✅ Safe (Whitelisted Domain)",
                                   description="This URL is from a highly trusted source.",
                                   reasons="• Domain is on a trusted whitelist<br>• No major phishing indicators found",
                                   original_input_url=user_input_url,
                                   final_display_url=final_display_url)
    except Exception as e:
        print(f"ERROR: Error during whitelist check for {actual_url_for_analysis}: {e}")
        traceback.print_exc()
        pass

    try:
        features_df, _ = extract_features(actual_url_for_analysis)

        if features_df.empty:
            raise ValueError("Feature extraction returned an empty DataFrame.")

        parsed_original_input_url = urlparse(user_input_url)
        is_original_input_shortened = any(shortener in parsed_original_input_url.netloc.lower() for shortener in [
            'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'shorte.st',
            'cutt.ly', 'is.gd', 'rebrand.ly', 'clck.ru', 'rb.gy'
        ])
        if 'is_shortened' in features_df.columns:
            features_df['is_shortened'] = 1 if is_original_input_shortened else 0


        try:
            expected_features = scaler.feature_names_in_
        except AttributeError:
            print("Warning: Scaler does not have 'feature_names_in_'. Relying on current feature_df columns order.")
            expected_features = features_df.columns.tolist()

        features_df_reindexed = features_df.reindex(columns=expected_features, fill_value=0)

        scaled_features = scaler.transform(features_df_reindexed)

        probs = model.predict_proba(scaled_features)[0]

        # --- CRITICAL FIX START (Corrected Logic) ---
        # Based on train_model.py, model.classes_ = [0, 1] where 0 is Phishing and 1 is Safe.
        # So, probs[0] is P(Phishing), and probs[1] is P(Safe).

        # A URL is classified as phishing if its probability of being phishing (probs[0])
        # exceeds the PHISHING_THRESHOLD.
        is_phishing = (probs[0] > PHISHING_THRESHOLD)

        if is_phishing:
            # If classified as phishing, confidence is based on the phishing probability (probs[0])
            confidence = round(probs[0] * 100, 2)
            result = f"🚨 Likely Phishing ({confidence}% confidence)"
            description = "⚠️ This URL seems suspicious. Be careful before opening it!"
        else:
            # If classified as safe, confidence is based on the safe probability (probs[1])
            confidence = round(probs[1] * 100, 2)
            result = f"✅ Safe ({confidence}% confidence)"
            description = "This URL looks clean and safe to access."
        # --- CRITICAL FIX END ---

        print(f"\n--- Model Prediction Details for {user_input_url} ---")
        print(f"Raw Probabilities (Class 0: {model.classes_[0]}, Class 1: {model.classes_[1]}): {probs}")
        print(f"PHISHING_THRESHOLD used: {PHISHING_THRESHOLD}")
        print(f"Is Phishing (True/False): {is_phishing}")
        print(f"Calculated Confidence: {confidence}%")
        print("---------------------------------------------------\n")

        reasons_list = []
        if is_phishing:
            reasons_list.append("Detected by trained ML model")

            if features_df_reindexed['has_suspicious_words'].iloc[0] == 1:
                reasons_list.append("Suspicious keywords found in URL path or query")
            if features_df_reindexed['has_suspicious_domain_keywords'].iloc[0] == 1:
                reasons_list.append("Suspicious keywords found in domain or hostname")
            if features_df_reindexed['has_ip'].iloc[0] == 1:
                reasons_list.append("IP address found in hostname (instead of domain name)")
            if features_df_reindexed['has_at_symbol'].iloc[0] == 1:
                reasons_list.append("Presence of '@' symbol (often used to obscure true URL)")
            if features_df_reindexed['is_shortened'].iloc[0] == 1:
                reasons_list.append("Uses a URL shortening service (original input was shortened)")
            if features_df_reindexed['has_https'].iloc[0] == 0:
                reasons_list.append("Lacks HTTPS (common for phishing on sensitive sites)")
            if features_df_reindexed['punycode_encoded'].iloc[0] == 1:
                reasons_list.append("Punycode encoded (often used for homograph attacks)")
            if features_df_reindexed['url_length'].iloc[0] > 75:
                reasons_list.append("URL is unusually long (after unshortening if applicable)")
            if features_df_reindexed['num_dots'].iloc[0] > 4:
                reasons_list.append("Unusual number of dots in URL (can indicate deception)")
            if features_df_reindexed['num_subdomains'].iloc[0] > 2:
                reasons_list.append("Many subdomains present (can indicate deceptive structure)")
            if features_df_reindexed['num_digits'].iloc[0] > 5:
                reasons_list.append("High number of digits in URL (can be suspicious)")
            if features_df_reindexed['num_consecutive_digits'].iloc[0] == 1:
                reasons_list.append("Contains consecutive digits (often in generated phishing domains)")

            if len(reasons_list) <= 1:
                reasons_list.append("Presence of risky patterns or unusual characteristics")

            reasons_list.append("Manual verification strongly recommended")

            reasons = "<br>".join([f"• {r}" for r in reasons_list])

        else: # If not phishing, default safe reasons
            reasons = """
                • No major phishing indicators found<br>
                • Follows common safe URL patterns<br>
                • Still, proceed with basic caution
            """

        return render_template("index.html",
                               result=result,
                               description=description,
                               reasons=reasons,
                               original_input_url=user_input_url,
                               final_display_url=final_display_url)

    except Exception as e:
        print(f"ERROR: Error during analysis: {e}")
        traceback.print_exc()
        return render_template("index.html",
                               result="❌ Error Occurred",
                               description=f"An error occurred while analyzing the URL: {e}. Please check server logs for details.",
                               reasons="Check the URL format or try again later.",
                               original_input_url=user_input_url,
                               final_display_url="")


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True) # Ensure host is 0.0.0.0 for external access

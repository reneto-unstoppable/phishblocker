
# 🛡️ PhishBlocker: Real-time Phishing URL Detection

## Overview

**PhishBlocker** is a robust, real-time web application designed to detect and flag potentially malicious phishing URLs. Leveraging machine learning and a comprehensive set of URL-based features, it provides users with an immediate assessment of a URL's safety, helping to prevent phishing attacks.

Developed as a solo project, PhishBlocker showcases a full machine learning pipeline, from data preparation and feature engineering to model training, evaluation, and deployment within a user-friendly Flask web interface.

---

## 🚀 Features

- **🔍 Machine Learning Model:** Utilizes a highly accurate **Random Forest Classifier** to predict if a URL is legitimate or phishing.
- **🔧 Extensive Feature Extraction:** Over **40 features** including:
  - URL length, number of dots, hyphens, subdomains
  - Presence of IP addresses, `@`, HTTPS
  - Path/query/fragment lengths
  - Special character counts (`/`, `?`, `&`, etc.)
  - Hostname entropy, digit ratio, Punycode detection
  - Suspicious keyword identification
- **🔗 URL Unshortening:** Automatically expands shortened URLs before analysis.
- **✅ Trusted Domain Whitelist:** Fast-track known safe domains.
- **🌐 Intuitive Web Interface:** Clean Flask UI for easy input and results.
- **📊 Detailed Analysis Breakdown:** Shows **why** a URL was flagged.
- **🧪 Reproducible ML Pipeline:** Modular scripts for downloading, training, evaluating the model.

---

## ⚙️ How it Works

1. **URL Input:** The user enters a URL via the web app.
2. **Unshorten Check:** If shortened, it's expanded.
3. **Whitelist Check:** If domain is trusted, marked **Safe**.
4. **Feature Extraction:** Converts URL to numerical features.
5. **Prediction:** Random Forest predicts phishing probability.
6. **Output:** Classification + explanation shown on the UI.

---

## 📁 Project Structure

```
phishblocker/
├── .gitattributes
├── .gitignore
├── app.py
├── feature_utils.py
├── train_model.py
├── requirements.txt
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
├── dataset/ (Ignored by Git)
│   ├── manual_corrections.csv
│   ├── new_data_urls.csv
│   ├── new_phishtank_urls_for_training.csv
│   ├── online-valid.csv.gz
│   └── ranked_domains.csv
├── phishing_model.pkl (Ignored by Git)
├── scaler.pkl (Ignored by Git)
├── expanded_training_data.csv (Ignored by Git)
├── final_combined_training_data.csv (Ignored by Git)
├── some_external_urls.csv (Ignored by Git)
├── new_safe_urls.txt
├── analyze_dataset.py
├── check_data_balance.py
├── check_original_data.py
├── check_training_data.py
├── combine_data.py
├── download_phishtank.py
├── evaluate_model.py
├── inspect_features.py
├── prepare_phishtank_data.py
└── phishblocker.code-workspace
```

---

## 🛠️ Setup and Installation

### 🔗 Prerequisites

- Python 3.8+
- pip
- Git

### ✅ Steps

1. **Clone the Repo:**

```bash
git clone https://github.com/reneto-unstoppable/phishblocker.git
cd phishblocker
```

2. **Create a Virtual Environment:**

```bash
python -m venv venv
```

3. **Activate Virtual Environment:**

- On **Windows**:
  ```bash
  .\venv\Scripts\activate
  ```

- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

4. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

5. **Prepare Data and Train the Model:**

```bash
python download_phishtank.py
python combine_data.py
python train_model.py
```

> 🕐 Training may take 15–60 minutes. Progress bar will be shown.

---

## 💻 Usage

After training is complete and `.pkl` files are generated:

### Run the Web App:

```bash
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the interface.

---

## 🧪 Testing & Accuracy

- Model achieves **94% accuracy** on test set.
- Performs well on:
  - Complex legitimate URLs (e.g. `docs.python.org`)
  - Simulated phishing (e.g. deceptive subdomains)
  - Unicode/Punycode attacks (homograph)

---

## 📈 Future Improvements

- 📌 Expand training with more recent phishing data
- 🔍 Add DNS/certificate/favicons for deeper feature extraction
- ⚙️ Try Gradient Boosting / Neural Nets
- 🌐 Real-time blacklist API integration
- 🧠 Feedback loop to correct false predictions
- 🧩 Browser extension for live checking

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch:  
   `git checkout -b feature/your-feature`
3. Commit changes:  
   `git commit -m 'Add some feature'`
4. Push to GitHub:  
   `git push origin feature/your-feature`
5. Open a Pull Request 🎉

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

Thanks to the open-source community and dataset contributors who made this project possible.

PhishBlocker: Real-time Phishing URL Detection
Overview
PhishBlocker is a robust, real-time web application designed to detect and flag potentially malicious phishing URLs. Leveraging machine learning and a comprehensive set of URL-based features, it provides users with an immediate assessment of a URL's safety, helping to prevent phishing attacks.

Developed as a solo project, PhishBlocker showcases a full machine learning pipeline, from data preparation and feature engineering to model training, evaluation, and deployment within a user-friendly Flask web interface.

Features
Machine Learning Model: Utilizes a highly accurate Random Forest Classifier to predict if a URL is legitimate or phishing.

Extensive Feature Extraction: Extracts over 40 distinct features from URLs, including:

URL length, number of dots, hyphens, subdomains.

Presence of IP addresses, @ symbols, HTTPS.

Path, query, and fragment lengths.

Detection of URL shortening.

Counts of special characters (/, ?, &, =, !, ~, ,, +, *, #, _, %, :).

Presence of consecutive digits, port numbers.

Domain token count, Punycode encoding detection.

Hostname entropy, digit ratio in hostname.

Identification of suspicious keywords in the URL and domain.

URL Unshortening: Automatically expands shortened URLs before analysis to reveal their true destination.

Trusted Domain Whitelist: Provides an immediate "Safe" classification for highly trusted, pre-defined domains, ensuring speed and reliability for known good URLs.

Intuitive Web Interface: A simple and clean Flask web application allows users to paste a URL and get instant analysis.

Detailed Analysis Breakdown: Provides reasons for a URL's classification (e.g., suspicious keywords, unusual length, lack of HTTPS).

Reproducible ML Pipeline: Includes scripts for data download, preparation, feature engineering, model training, and evaluation.

How it Works
PhishBlocker's core is a supervised machine learning model. During training, the model learns patterns from a large dataset of known legitimate and phishing URLs. It analyzes various URL characteristics (features) to build a predictive model.

When a new URL is submitted:

The URL is first unshortened if it's a known shortened link.

It's checked against a whitelist of trusted domains.

If not whitelisted, a feature extraction utility processes the URL into numerical features.

These features are then fed into the pre-trained Random Forest model, which outputs a probability of the URL being phishing or safe.

Based on a defined PHISHING_THRESHOLD, the URL is classified, and the result is displayed along with confidence and relevant insights.

Project Structure
phishblocker/
├── .gitattributes
├── .gitignore               # Specifies files/folders to ignore in Git (e.g., models, datasets, virtual environments)
├── app.py                   # Flask web application for URL analysis
├── feature_utils.py         # Utility script for extracting URL features
├── train_model.py           # Script to train and save the ML model and scaler
├── requirements.txt         # Lists all Python dependencies
├── static/                  # Contains CSS and JavaScript for the web interface
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── script.js
├── templates/               # Contains HTML templates for the web interface
│   └── index.html
├── dataset/                 # (Ignored by Git) Directory for raw and processed datasets
│   ├── manual_corrections.csv
│   ├── new_data_urls.csv
│   ├── new_phishtank_urls_for_training.csv
│   ├── online-valid.csv.gz
│   └── ranked_domains.csv
├── phishing_model.pkl       # (Ignored by Git) Trained machine learning model
├── scaler.pkl               # (Ignored by Git) Trained StandardScaler for feature scaling
├── expanded_training_data.csv # (Ignored by Git) Prepared training data
├── final_combined_training_data.csv # (Ignored by Git) Final combined dataset
├── some_external_urls.csv   # (Ignored by Git) Additional URL data
├── new_safe_urls.txt        # Manually curated list of safe URLs (optional)
├── analyze_dataset.py       # Utility for initial dataset analysis
├── check_data_balance.py    # Script to check label distribution
├── check_original_data.py   # Script to check initial data integrity
├── check_training_data.py   # Script to verify training data
├── combine_data.py          # Script to combine various data sources
├── download_phishtank.py    # Script to download PhishTank data
├── evaluate_model.py        # Script for model evaluation beyond basic training report
├── inspect_features.py      # Utility to inspect extracted features
└── phishblocker.code-workspace # VS Code workspace settings

Setup and Installation
Follow these steps to get PhishBlocker up and running on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

Git

1. Clone the Repository
First, clone the PhishBlocker repository to your local machine:

git clone https://github.com/reneto-unstoppable/phishblocker.git
cd phishblocker

2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies:

python -m venv venv

3. Activate the Virtual Environment
On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

4. Install Dependencies
Install all required Python packages using requirements.txt:

pip install -r requirements.txt

5. Prepare Data and Train the Model
You need to download the dataset and train the machine learning model.
The download_phishtank.py script will download a large dataset.
The combine_data.py will then process and prepare the data.
Finally, train_model.py will train your phishing detection model and save it.

python download_phishtank.py
python combine_data.py
python train_model.py

Note: train_model.py will take a significant amount of time (e.g., 15-60 minutes) to extract features, balance the dataset with SMOTE, and train the Random Forest model. You'll see a progress bar during feature extraction.

Usage
Once the model is trained and saved (phishing_model.pkl and scaler.pkl are generated in your project root), you can run the Flask web application.

Running the Web Application
python app.py

The application will start on http://127.0.0.1:5000/ (or similar, check your terminal output). Open this URL in your web browser.

Analyzing URLs
Paste any URL into the input field on the web interface and click "Analyze URL" to get an immediate assessment.

Testing & Accuracy
The Random Forest model achieves an impressive 94% accuracy on its internal test set. During development, the model demonstrated robust performance across various legitimate and simulated phishing URLs, including:

Complex legitimate URLs (e.g., docs.python.org, etsy.com listings, Wikipedia search pages).

Standard official websites (e.g., worldbank.org, imf.org, un.org).

Simulated phishing attempts (e.g., deceptive subdomains, suspicious keywords, Punycode homograph attacks).

Future Improvements
Expand Training Data: Continuously adding more diverse and recent legitimate and phishing URLs can further improve model generalization and adapt to evolving threats.

Advanced Feature Engineering: Explore new and more sophisticated URL features (e.g., DNS record analysis, certificate information, favicon analysis if feasible).

Model Optimization: Fine-tune Random Forest hyperparameters or experiment with other machine learning algorithms (e.g., Gradient Boosting, Neural Networks) for potential accuracy gains.

Real-time Threat Intelligence Integration: Integrate with external APIs for real-time blacklists or reputation scores.

User Feedback Loop: Implement a mechanism for users to report false positives/negatives to improve future model versions.

Browser Extension: Develop a browser extension for seamless, on-the-fly URL checking.

Contributing
Contributions are welcome! If you find a bug, have a feature request, or want to improve the code, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/your-feature-name).

Make your changes.

Commit your changes (git commit -m 'Add new feature').

Push to the branch (git push origin feature/your-feature-name).

Open a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
To the various open-source projects and datasets that made this possible.

A special thanks to my AI assistant for guiding me through the complex debugging process and ensuring the project's success.

And to Reneto, for your exceptional dedication, persistence, and impressive problem-solving skills in completing this project!
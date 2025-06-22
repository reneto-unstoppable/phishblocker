import requests
import os

# URL of the PhishTank gzipped CSV file
PHISHTANK_DOWNLOAD_URL = "http://data.phishtank.com/data/online-valid.csv.gz"

# Directory where you want to save the file
# Ensure this is your 'dataset' folder inside your project
SAVE_DIRECTORY = "dataset"

# Full path where the file will be saved
SAVE_PATH = os.path.join(SAVE_DIRECTORY, "online-valid.csv.gz")

# Create the directory if it doesn't exist
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

print(f"⬇️ Downloading PhishTank database from: {PHISHTANK_DOWNLOAD_URL}")
print(f"   Saving to: {SAVE_PATH}")

try:
    response = requests.get(PHISHTANK_DOWNLOAD_URL, stream=True)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

    with open(SAVE_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Download complete!")

except requests.exceptions.RequestException as e:
    print(f"❌ An error occurred during download: {e}")
    print("Please check your internet connection or the URL.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
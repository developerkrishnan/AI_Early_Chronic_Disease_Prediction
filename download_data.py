import os
import urllib.request as request

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

datasets = {
    "diabetes.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "heart.csv": "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
    "kidney.csv": "https://raw.githubusercontent.com/ArjunAnilPillai/Chronic-Kidney-Disease-dataset/master/final.csv"
}

def download_file(name, url):
    print(f"Downloading {name}...")
    try:
        file_path = os.path.join(DATA_DIR, name)
        request.urlretrieve(url, file_path)
        print(f"Successfully saved {name} to {file_path}")
    except Exception as e:
        print(f"Error downloading {name}: {e}")

if __name__ == "__main__":
    for name, url in datasets.items():
        download_file(name, url)
    print("All downloads processed.")

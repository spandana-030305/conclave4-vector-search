import os
import tarfile
import requests
import pandas as pd

# Paths
CSV_PATH = r"D:\conclave4-vector-search\data\pmc_articles.csv"
TAR_DIR = r"D:\conclave4-vector-search\data\tar"
XML_DIR = r"D:\conclave4-vector-search\data\xml"

# Create folders
os.makedirs(TAR_DIR, exist_ok=True)
os.makedirs(XML_DIR, exist_ok=True)

# Load metadata
df = pd.read_csv(CSV_PATH)

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"

def download_and_extract(row):
    rel_path = row["File"]  # e.g. oa_package/PMC1234567.tar.gz
    url = BASE_URL + rel_path

    tar_name = os.path.basename(rel_path)
    tar_path = os.path.join(TAR_DIR, tar_name)

    # Download tar.gz
    if not os.path.exists(tar_path):
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract XML only
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".xml"):
                tar.extract(member, XML_DIR)

# Process files safely (limit speed)
for idx, row in df.iterrows():
    try:
        download_and_extract(row)
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} / {len(df)}")
    except Exception as e:
        print(f"Failed at index {idx}: {e}")

print("Download & XML extraction completed")



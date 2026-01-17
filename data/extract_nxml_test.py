import os
import tarfile

TAR_DIR = r"D:\conclave4-vector-search\data\tar"
NXML_DIR = r"D:\conclave4-vector-search\data\nxml_test"

os.makedirs(NXML_DIR, exist_ok=True)

# Get list of .tar.gz files
tar_files = [f for f in os.listdir(TAR_DIR) if f.endswith(".tar.gz")]

# Take only first 2
tar_files = tar_files[:2]

print("Testing extraction on files:")
for f in tar_files:
    print(" -", f)

count = 0

for tar_name in tar_files:
    tar_path = os.path.join(TAR_DIR, tar_name)

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".nxml"):
                    member.name = os.path.basename(member.name)  # flatten
                    tar.extract(member, NXML_DIR)
                    count += 1
    except Exception as e:
        print(f"❌ Failed on {tar_name}: {e}")

print(f"Extracted {count} .nxml files (test run)")

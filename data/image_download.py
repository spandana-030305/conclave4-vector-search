import os
import numpy as np
from medmnist import ChestMNIST, PathMNIST
from torchvision import transforms
from PIL import Image

# =========================
# CONFIG
# =========================
OUTPUT_DIR = "data/images"
MAX_IMAGES_PER_DATASET = 200

DATASETS = {
    "chestmnist": ChestMNIST,
    "pathmnist": PathMNIST
}

# =========================
# CREATE FOLDERS
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToTensor()
])

# =========================
# DOWNLOAD & SAVE
# =========================
for dataset_name, DatasetClass in DATASETS.items():
    print(f"\nDownloading {dataset_name}...")

    dataset = DatasetClass(
        split="train",
        transform=transform,
        download=True
    )

    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    count = 0

    for i in range(len(dataset)):
        if count >= MAX_IMAGES_PER_DATASET:
            break

        image_tensor, label = dataset[i]

        # Convert tensor to numpy
        image_np = image_tensor.numpy()

        # Case 1: Grayscale (1, H, W)
        if image_np.ndim == 3 and image_np.shape[0] == 1:
            image_np = image_np[0]  # (H, W)
            image = Image.fromarray((image_np * 255).astype(np.uint8))
            image = image.convert("RGB")

        # Case 2: RGB (3, H, W)
        elif image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)  # (H, W, 3)
            image = Image.fromarray((image_np * 255).astype(np.uint8))

        else:
            print(f"Skipping image with shape {image_np.shape}")
            continue

        image_path = os.path.join(dataset_dir, f"{dataset_name}_{count}.png")
        image.save(image_path)

        count += 1

    print(f"Saved {count} images to {dataset_dir}")

print("\nImage download completed successfully")


import random
import shutil
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
SOURCE_DIR = Path("data/radiography_raw")

OUTPUT_DIR = Path("data/radiography")

CLASSES = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia",
]

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

SEED = 42

random.seed(SEED)

# -----------------------------
# Create folders
# -----------------------------
for split in ["train", "val", "test"]:

    for cls in CLASSES:

        path = OUTPUT_DIR / split / cls

        path.mkdir(
            parents=True,
            exist_ok=True
        )

# -----------------------------
# Copy files
# -----------------------------
for cls in CLASSES:

    image_dir = SOURCE_DIR / cls / "images"

    images = list(image_dir.glob("*"))

    images = [
        img for img in images
        if img.suffix.lower()
        in [".png", ".jpg", ".jpeg"]
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_SPLIT)
    val_end = int(total * (TRAIN_SPLIT + VAL_SPLIT))

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    for split_name, split_images in splits.items():

        for img_path in split_images:

            destination = (
                OUTPUT_DIR
                / split_name
                / cls
                / img_path.name
            )

            shutil.copy2(
                img_path,
                destination
            )

    print(
        f"{cls}: "
        f"{len(train_images)} train | "
        f"{len(val_images)} val | "
        f"{len(test_images)} test"
    )

print("\nDataset prepared successfully.")
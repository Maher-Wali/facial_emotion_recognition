import os
import csv
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
FERPLUS    = os.path.join(DATA_DIR, "ferplus")
RAFDB      = os.path.join(DATA_DIR, "rafdb", "DATASET")
OUT_DIR    = os.path.join(DATA_DIR, "exploration")
os.makedirs(OUT_DIR, exist_ok=True)

# ── RAF-DB label mapping (official) ───────────────────────────────────────────
RAFDB_LABELS = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happy",
    "5": "Sad",
    "6": "Angry",
    "7": "Neutral",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def count_images_in_dir(root):
    """Return {class_name: count} by walking one level of subfolders."""
    counts = {}
    for cls in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        counts[cls] = len(imgs)
    return counts

def sample_images(root, class_name, n=1):
    """Return up to n image paths from a class folder."""
    cls_path = os.path.join(root, class_name)
    files = [f for f in os.listdir(cls_path)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return [os.path.join(cls_path, f) for f in files[:n]]

def image_stats(img_path):
    """Return (width, height, mode) for an image."""
    with Image.open(img_path) as img:
        return img.size[0], img.size[1], img.mode

def plot_distribution(counts, title, color, out_path):
    labels = list(counts.keys())
    values = list(counts.values())
    total  = sum(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=color, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Emotion", fontsize=11)
    ax.set_ylabel("Number of images", fontsize=11)
    ax.set_ylim(0, max(values) * 1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val}\n({val/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=8.5)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

def plot_samples(dataset_root, classes, title, out_path, label_map=None):
    """Save a grid of one sample per class."""
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2.8))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, cls in zip(axes, classes):
        paths = sample_images(dataset_root, cls)
        if not paths:
            ax.axis("off")
            continue
        with Image.open(paths[0]) as img:
            ax.imshow(img, cmap="gray" if img.mode == "L" else None)
        display_label = label_map.get(cls, cls) if label_map else cls
        ax.set_title(display_label, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ── FERplus exploration ────────────────────────────────────────────────────────
print("\n" + "="*30)
print(" FERplus")
print("="*30)

ferplus_splits = {}
for split in ["train", "validation", "test"]:
    split_path = os.path.join(FERPLUS, split)
    counts = count_images_in_dir(split_path)
    ferplus_splits[split] = counts
    total = sum(counts.values())
    print(f"\n  {split.upper()} — {total} images")
    for cls, n in counts.items():
        print(f"    {cls:<12} {n:>5}  ({n/total*100:.1f}%)")

# Sample image stats from train
train_angry = sample_images(os.path.join(FERPLUS, "train"), "angry")
if train_angry:
    w, h, mode = image_stats(train_angry[0])
    print(f"\n  Image format : {mode}, {w}×{h} px")

# Distribution plot (train)
plot_distribution(
    ferplus_splits["train"],
    "FERplus — Training set class distribution",
    "#4C72B0",
    os.path.join(OUT_DIR, "ferplus_train_distribution.png"),
)

# Sample grid
plot_samples(
    os.path.join(FERPLUS, "train"),
    list(ferplus_splits["train"].keys()),
    "FERplus — one sample per class (train)",
    os.path.join(OUT_DIR, "ferplus_samples.png"),
)


# ── RAF-DB exploration ─────────────────────────────────────────────────────────
print("\n" + "="*30)
print(" RAF-DB")
print("="*30)

rafdb_splits = {}
for split in ["train", "test"]:
    split_path = os.path.join(RAFDB, split)
    counts_raw = count_images_in_dir(split_path)
    # Map numeric folder names to emotion labels
    counts = {RAFDB_LABELS.get(k, k): v for k, v in counts_raw.items()}
    rafdb_splits[split] = counts
    total = sum(counts.values())
    print(f"\n  {split.upper()} — {total} images")
    for cls, n in counts.items():
        print(f"    {cls:<12} {n:>5}  ({n/total*100:.1f}%)")

# Sample image stats
rafdb_folder_1 = os.path.join(RAFDB, "train", "1")
sample = [os.path.join(rafdb_folder_1, f)
          for f in os.listdir(rafdb_folder_1)
          if f.lower().endswith((".jpg", ".jpeg", ".png"))][0]
w, h, mode = image_stats(sample)
print(f"\n  Image format : {mode}, {w}×{h} px (variable sizes)")

# Distribution plot (train)
plot_distribution(
    rafdb_splits["train"],
    "RAF-DB — Training set class distribution",
    "#DD8452",
    os.path.join(OUT_DIR, "rafdb_train_distribution.png"),
)

# Sample grid (use numeric folder names, pass label map)
numeric_classes = sorted(os.listdir(os.path.join(RAFDB, "train")))
plot_samples(
    os.path.join(RAFDB, "train"),
    numeric_classes,
    "RAF-DB — one sample per class (train)",
    os.path.join(OUT_DIR, "rafdb_samples.png"),
    label_map=RAFDB_LABELS,
)

print("\nExploration complete. Outputs in:", OUT_DIR)

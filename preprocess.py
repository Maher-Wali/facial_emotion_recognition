import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_SIZE  = (112, 112)   # output resolution for both datasets
CLAHE_CLIP   = 2.0          # CLAHE clip limit
CLAHE_GRID   = (8, 8)       # CLAHE tile grid size

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
FERPLUS_IN   = os.path.join(DATA_DIR, "ferplus")
RAFDB_IN     = os.path.join(DATA_DIR, "rafdb", "DATASET")
FERPLUS_OUT  = os.path.join(DATA_DIR, "ferplus_processed")
RAFDB_OUT    = os.path.join(DATA_DIR, "rafdb_processed")
ILLUS_DIR    = os.path.join(DATA_DIR, "illustrations")

RAFDB_LABELS = {
    "1": "Surprise", "2": "Fear",    "3": "Disgust",
    "4": "Happy",    "5": "Sad",     "6": "Angry",
    "7": "Neutral",
}

# FERplus folder typo fix
FERPLUS_RENAME = {"suprise": "surprise"}

# ── Core transforms ────────────────────────────────────────────────────────────
def load_gray(path):
    """Load any image as uint8 grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"Cannot read: {path}")
    return img

def resize(img, size=TARGET_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(img)

def normalize(img):
    """Return float32 array in [0, 1]."""
    return img.astype(np.float32) / 255.0

def pipeline(path):
    """Full preprocessing pipeline: load -> resize -> CLAHE -> normalize."""
    img = load_gray(path)
    img = resize(img)
    img = apply_clahe(img)
    return normalize(img)

def save_processed(arr, dest_path):
    """Save normalized float32 array back as uint8 PNG."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cv2.imwrite(dest_path, (arr * 255).astype(np.uint8))


# ── Illustration helpers ───────────────────────────────────────────────────────
def make_step_row(path):
    """Return list of (label, uint8 image) for each pipeline step."""
    raw   = load_gray(path)
    rsz   = resize(raw)
    clahe = apply_clahe(rsz)
    norm  = (normalize(clahe) * 255).astype(np.uint8)
    return [
        ("Original",  raw),
        ("Resized\n112x112", rsz),
        ("CLAHE",     clahe),
        ("Normalized", norm),
    ]

def save_illustration(samples, title, out_path, cmap="gray"):
    """
    samples: list of (emotion_label, path)
    Shows the 4-step pipeline for each sample emotion.
    """
    steps = [make_step_row(p) for _, p in samples]
    n_emotions = len(samples)
    n_steps    = len(steps[0])

    fig, axes = plt.subplots(
        n_emotions, n_steps,
        figsize=(n_steps * 2.4, n_emotions * 2.6)
    )
    if n_emotions == 1:
        axes = [axes]

    for row_idx, ((emotion, _), step_row) in enumerate(zip(samples, steps)):
        for col_idx, (step_label, img) in enumerate(step_row):
            ax = axes[row_idx][col_idx]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(step_label, fontsize=9, fontweight="bold", pad=4)
            if col_idx == 0:
                ax.set_ylabel(emotion, fontsize=9, rotation=0,
                              labelpad=50, va="center")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved illustration: {out_path}")


# ── Process FERplus ────────────────────────────────────────────────────────────
def process_ferplus():
    print("\n" + "="*40)
    print(" Processing FERplus")
    print("="*40)

    splits = ["train", "validation", "test"]
    stats  = {"ok": 0, "skipped": 0}
    illus_samples = []   # (emotion, path) for illustration

    for split in splits:
        split_in  = os.path.join(FERPLUS_IN, split)
        split_out = os.path.join(FERPLUS_OUT, split)

        for cls_folder in sorted(os.listdir(split_in)):
            cls_in  = os.path.join(split_in, cls_folder)
            if not os.path.isdir(cls_in):
                continue

            # Fix typo in folder name
            cls_name = FERPLUS_RENAME.get(cls_folder, cls_folder)
            cls_out  = os.path.join(split_out, cls_name)
            os.makedirs(cls_out, exist_ok=True)

            for fname in os.listdir(cls_in):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                src  = os.path.join(cls_in, fname)
                dest = os.path.join(cls_out, os.path.splitext(fname)[0] + ".png")
                try:
                    arr = pipeline(src)
                    save_processed(arr, dest)
                    stats["ok"] += 1
                except OSError:
                    stats["skipped"] += 1

            # Collect one sample per class from train for illustration
            if split == "train" and len(illus_samples) < 4:
                files = [f for f in os.listdir(cls_in)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if files:
                    illus_samples.append((cls_name, os.path.join(cls_in, files[0])))

        print(f"  {split}: done")

    print(f"  Total processed : {stats['ok']}  |  Skipped (corrupt): {stats['skipped']}")

    # Generate illustration
    save_illustration(
        illus_samples,
        "FERplus - Preprocessing pipeline (4 sample emotions)",
        os.path.join(ILLUS_DIR, "ferplus_before_after.png"),
    )


# ── Process RAF-DB ─────────────────────────────────────────────────────────────
def process_rafdb():
    print("\n" + "="*40)
    print(" Processing RAF-DB")
    print("="*40)

    splits = ["train", "test"]
    stats  = {"ok": 0, "skipped": 0}
    illus_samples = []

    for split in splits:
        split_in  = os.path.join(RAFDB_IN,  split)
        split_out = os.path.join(RAFDB_OUT, split)

        for num_folder in sorted(os.listdir(split_in)):
            cls_in = os.path.join(split_in, num_folder)
            if not os.path.isdir(cls_in):
                continue

            cls_name = RAFDB_LABELS.get(num_folder, num_folder)
            cls_out  = os.path.join(split_out, cls_name)
            os.makedirs(cls_out, exist_ok=True)

            for fname in os.listdir(cls_in):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                src  = os.path.join(cls_in, fname)
                dest = os.path.join(cls_out, os.path.splitext(fname)[0] + ".png")
                try:
                    arr = pipeline(src)
                    save_processed(arr, dest)
                    stats["ok"] += 1
                except OSError:
                    stats["skipped"] += 1

            if split == "train" and len(illus_samples) < 4:
                files = [f for f in os.listdir(cls_in)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if files:
                    illus_samples.append((cls_name, os.path.join(cls_in, files[0])))

        print(f"  {split}: done")

    print(f"  Total processed : {stats['ok']}  |  Skipped (corrupt): {stats['skipped']}")

    save_illustration(
        illus_samples,
        "RAF-DB - Preprocessing pipeline (4 sample emotions)",
        os.path.join(ILLUS_DIR, "rafdb_before_after.png"),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(ILLUS_DIR, exist_ok=True)
    process_ferplus()
    process_rafdb()
    print("\nPreprocessing complete.")
    print(f"  FERplus processed : {FERPLUS_OUT}")
    print(f"  RAF-DB  processed : {RAFDB_OUT}")
    print(f"  Illustrations     : {ILLUS_DIR}")

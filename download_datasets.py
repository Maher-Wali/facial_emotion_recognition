import os
import shutil
import kagglehub

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASETS = {
    "ferplus": "arnabkumarroy02/ferplus",
    "rafdb":   "shuvoalok/raf-db-dataset",   # verify slug on Kaggle before running
}

def download(name, slug, dest):
    print(f"\n[{name}] Downloading '{slug}' ...")
    cache_path = kagglehub.dataset_download(slug)
    print(f"[{name}] Cached at: {cache_path}")

    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(cache_path, dest)
    print(f"[{name}] Copied to: {dest}")

    # Quick sanity check
    entries = os.listdir(dest)
    print(f"[{name}] Top-level contents: {entries}")

if __name__ == "__main__":
    for name, slug in DATASETS.items():
        dest = os.path.join(DATA_DIR, name)
        download(name, slug, dest)

    print("\nAll datasets ready.")

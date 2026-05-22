"""Extract ImageNet class indices from BigGAN-ai and SDv1.5-ai filenames,
compute intersection, sort by min count, and print with class names.

The AI filenames have the format <class_idx_3digit>_<generator>_<sample_idx>.png
where class_idx is the ImageNet-1k class index (0-999).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

AI_CLASS_RE = re.compile(r"/(\d{3})_(?:biggan|sdv5|adm|midjourney)_\d+\.png", re.IGNORECASE)

IMAGENET_CLASS_INDEX_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)


def load_hf_token() -> str:
    project_root = Path(__file__).resolve().parents[2]
    experiment_root = Path(__file__).resolve().parents[1]
    for env_path in (experiment_root / ".env", project_root / ".env"):
        if env_path.is_file():
            load_dotenv(env_path, override=False)
    token = os.environ.get("HF_TOKEN", "").strip().strip('"').strip("'")
    if not token:
        raise RuntimeError("HF_TOKEN missing")
    return token


def download_imagenet_idx_to_name() -> list[str]:
    """Download ImageNet-1k idx -> class name (1000 lines, idx 0..999)."""
    cache = Path(__file__).resolve().parents[1] / "results" / "imagenet_classes.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file():
        print(f"downloading {IMAGENET_CLASS_INDEX_URL}")
        with urllib.request.urlopen(IMAGENET_CLASS_INDEX_URL, timeout=30) as r:
            cache.write_bytes(r.read())
    names = cache.read_text(encoding="utf-8").splitlines()
    assert len(names) == 1000, f"expected 1000 names, got {len(names)}"
    return names


def extract_ai_class_idx(file_path: str) -> int | None:
    m = AI_CLASS_RE.search(file_path)
    return int(m.group(1)) if m else None


def main() -> None:
    token = load_hf_token()

    from huggingface_hub import hf_hub_download
    import pandas as pd

    names = download_imagenet_idx_to_name()

    meta_path = hf_hub_download(
        repo_id="Qwerty0193/genimage-processed",
        filename="metadata.parquet",
        repo_type="dataset",
        token=token,
    )
    df = pd.read_parquet(meta_path)

    biggan_ai = df[(df["generator"] == "BigGAN") & (df["label_name"] == "ai")].copy()
    sd_ai = df[(df["generator"] == "Stable Diffusion V1.5") & (df["label_name"] == "ai")].copy()
    biggan_ai["class_idx"] = biggan_ai["file_path"].map(extract_ai_class_idx)
    sd_ai["class_idx"] = sd_ai["file_path"].map(extract_ai_class_idx)

    print(f"BigGAN-ai: {len(biggan_ai)} rows, "
          f"{biggan_ai['class_idx'].notna().sum()} class_idx extracted, "
          f"{biggan_ai['class_idx'].dropna().nunique()} unique classes")
    print(f"SDv1.5-ai: {len(sd_ai)} rows, "
          f"{sd_ai['class_idx'].notna().sum()} class_idx extracted, "
          f"{sd_ai['class_idx'].dropna().nunique()} unique classes")

    bg_counts = biggan_ai["class_idx"].dropna().astype(int).value_counts()
    sd_counts = sd_ai["class_idx"].dropna().astype(int).value_counts()
    common = set(bg_counts.index) & set(sd_counts.index)
    print(f"\nclasses in BOTH BigGAN-ai and SDv1.5-ai: {len(common)}")

    overlap = pd.DataFrame({"biggan_ai": bg_counts, "sdv1_5_ai": sd_counts}).fillna(0).astype(int)
    overlap = overlap[overlap.index.isin(common)].copy()
    overlap["min_count"] = overlap.min(axis=1)
    overlap["class_name"] = [names[i] for i in overlap.index]
    overlap = overlap.sort_values("min_count", ascending=False)

    print("\ntop 40 overlapping classes (idx -> name, biggan_n, sdv5_n, min_n):")
    for idx, row in overlap.head(40).iterrows():
        print(f"  {idx:3d}  {row['class_name']:35s}  biggan={row['biggan_ai']:3d}  sd={row['sdv1_5_ai']:3d}  min={row['min_count']:3d}")

    out_path = Path(__file__).resolve().parents[1] / "results" / "class_overlap.csv"
    overlap.to_csv(out_path)
    print(f"\nwrote {out_path} ({len(overlap)} classes)")


if __name__ == "__main__":
    main()

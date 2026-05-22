"""Find ImageNet synsets that are present in BOTH BigGAN-ai and SDv1.5-ai
splits of genimage-processed, with enough samples.

The class is encoded only in the filename, e.g.
  ./data/genimage_subset/BigGAN/train/ai/n01530575_<idx>.JPEG -> synset n01530575
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

SYNSET_RE = re.compile(r"(n\d{8})_")


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


def extract_synset(file_path: str) -> str | None:
    m = SYNSET_RE.search(file_path)
    return m.group(1) if m else None


def main() -> None:
    token = load_hf_token()
    from huggingface_hub import hf_hub_download
    import pandas as pd

    meta_path = hf_hub_download(
        repo_id="Qwerty0193/genimage-processed",
        filename="metadata.parquet",
        repo_type="dataset",
        token=token,
    )
    df = pd.read_parquet(meta_path)

    df["synset"] = df["file_path"].map(extract_synset)
    n_missing = df["synset"].isna().sum()
    print(f"rows total: {len(df)}, rows with synset extracted: {len(df) - n_missing}")

    biggan_ai = df[(df["generator"] == "BigGAN") & (df["label_name"] == "ai")]
    sd_ai = df[(df["generator"] == "Stable Diffusion V1.5") & (df["label_name"] == "ai")]
    biggan_nature = df[(df["generator"] == "BigGAN") & (df["label_name"] == "nature")]
    sd_nature = df[(df["generator"] == "Stable Diffusion V1.5") & (df["label_name"] == "nature")]

    print(f"\nBigGAN-ai: {len(biggan_ai)} rows, {biggan_ai['synset'].nunique()} synsets")
    print(f"SDv1.5-ai: {len(sd_ai)} rows, {sd_ai['synset'].nunique()} synsets")
    print(f"BigGAN-nature: {len(biggan_nature)} rows, {biggan_nature['synset'].nunique()} synsets")
    print(f"SDv1.5-nature: {len(sd_nature)} rows, {sd_nature['synset'].nunique()} synsets")

    bg_counts = biggan_ai["synset"].value_counts()
    sd_counts = sd_ai["synset"].value_counts()
    common = set(bg_counts.index) & set(sd_counts.index)
    print(f"\nsynsets in BOTH BigGAN-ai and SDv1.5-ai: {len(common)}")

    overlap = pd.DataFrame({
        "biggan_ai": bg_counts,
        "sdv1_5_ai": sd_counts,
    }).fillna(0).astype(int)
    overlap = overlap[overlap.index.isin(common)]
    overlap["min_count"] = overlap.min(axis=1)
    overlap = overlap.sort_values("min_count", ascending=False)
    print("\ntop 40 overlapping synsets by min(biggan_ai, sdv1_5_ai):")
    print(overlap.head(40).to_string())

    out_path = Path(__file__).resolve().parents[1] / "results" / "category_overlap.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

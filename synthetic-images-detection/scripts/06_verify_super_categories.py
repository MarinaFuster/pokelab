"""Verify that the super-category definitions yield a reasonable
number of samples per (source, super-cat) for BigGAN and SDv1.5.

Prints per-super-cat class names so we can sanity-check the assignment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from huggingface_hub import hf_hub_download

from pipeline import (
    SUPER_CATEGORIES,
    extract_ai_class_idx,
    extract_synset,
    imagenet_class_names,
    imagenet_synset_to_idx,
    load_hf_token,
    super_category_for_idx,
)


def main() -> None:
    token = load_hf_token()
    names = imagenet_class_names()
    syn_to_idx = imagenet_synset_to_idx()

    print("=== super-category contents ===")
    for cat, ids in SUPER_CATEGORIES.items():
        examples = ", ".join(names[i] for i in sorted(ids)[:6])
        print(f"  {cat:10s} ({len(ids):3d} classes): e.g. {examples}, ...")

    meta_path = hf_hub_download(
        repo_id="Qwerty0193/genimage-processed",
        filename="metadata.parquet",
        repo_type="dataset",
        token=token,
    )
    df = pd.read_parquet(meta_path)
    df["synset"] = df["file_path"].map(extract_synset)
    df["ai_class_idx"] = df["file_path"].map(extract_ai_class_idx)
    df["nature_class_idx"] = df["synset"].map(lambda s: syn_to_idx.get(s) if s else None)
    df["class_idx"] = df["ai_class_idx"].fillna(df["nature_class_idx"])
    df["super_cat"] = df["class_idx"].map(
        lambda x: super_category_for_idx(int(x)) if pd.notna(x) else None
    )

    print("\n=== sample counts per (generator, label_name, super_cat) ===")
    pivot = (
        df.dropna(subset=["super_cat"])
        .groupby(["generator", "label_name", "super_cat"])
        .size()
        .unstack(fill_value=0)
    )
    print(pivot.to_string())

    print("\n=== sample counts focused on BigGAN-ai and SDv1.5-ai ===")
    for gen in ["BigGAN", "Stable Diffusion V1.5"]:
        sub = df[(df["generator"] == gen) & (df["label_name"] == "ai")]
        sub = sub.dropna(subset=["super_cat"])
        counts = sub.groupby("super_cat").size()
        print(f"\n{gen}:")
        for cat in SUPER_CATEGORIES:
            print(f"  {cat:10s}  {counts.get(cat, 0):4d} images")

    out_path = Path(__file__).resolve().parents[1] / "results" / "supercat_counts.csv"
    pivot.to_csv(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

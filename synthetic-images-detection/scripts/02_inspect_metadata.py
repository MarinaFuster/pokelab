"""Download Qwerty0193/genimage-processed/metadata.parquet and inspect schema.

We do not load the full image zip yet — that comes later, after we know which
rows we want.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_hf_token() -> str:
    project_root = Path(__file__).resolve().parents[2]
    experiment_root = Path(__file__).resolve().parents[1]
    for env_path in (experiment_root / ".env", project_root / ".env"):
        if env_path.is_file():
            load_dotenv(env_path, override=False)
    token = os.environ.get("HF_TOKEN", "").strip().strip('"').strip("'")
    if not token:
        raise RuntimeError("HF_TOKEN missing")
    os.environ["HF_TOKEN"] = token
    return token


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
    print(f"[OK] downloaded metadata.parquet -> {meta_path}")

    df = pd.read_parquet(meta_path)
    print(f"\nshape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"dtypes:\n{df.dtypes}")

    print("\nfirst 5 rows:")
    print(df.head(5).to_string())

    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 50:
            vc = df[col].value_counts(dropna=False).head(40)
            print(f"\n[{col}] {nunique} unique values:")
            print(vc.to_string())
        else:
            print(f"\n[{col}] {nunique} unique values (too many to list); sample: {df[col].dropna().unique()[:5].tolist()}")

    if "generator" in df.columns and ("imagenet_class" in df.columns or "class_id" in df.columns or "class" in df.columns or "label_name" in df.columns):
        print("\n--- categories present in BigGAN AND SD v1.5 (both as fake rows) ---")
    print("\ndone")


if __name__ == "__main__":
    main()

"""Look at AI-image file paths to understand how class is encoded."""

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
    df = pd.read_parquet(meta_path)

    for gen in ["BigGAN", "Stable Diffusion V1.5", "ADM", "GLIDE", "VQDM", "Midjourney"]:
        for lbl in ["ai", "nature"]:
            sub = df[(df["generator"] == gen) & (df["label_name"] == lbl)]
            print(f"\n=== {gen} / {lbl} ({len(sub)} rows) — sample 10 paths ===")
            for p in sub["file_path"].head(10).tolist():
                print(f"  {p}")


if __name__ == "__main__":
    main()

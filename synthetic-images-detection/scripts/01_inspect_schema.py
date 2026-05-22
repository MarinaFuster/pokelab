"""Inspect the schema of the synthetic and real datasets.

Uses HfApi to list repo files (fast) and only does a tiny streaming read
to learn schema. The previous version called get_dataset_config_names()
which hangs forever on this dataset on Windows.
"""

from __future__ import annotations

import os
import sys
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
        raise RuntimeError(
            "HF_TOKEN is not set or is empty. "
            "Create a .env file with HF_TOKEN=<your_token> in either the "
            "project root or the synthetic-images-detection/ folder."
        )
    os.environ["HF_TOKEN"] = token
    return token


def main() -> None:
    token = load_hf_token()
    print(f"[OK] HF_TOKEN loaded (len={len(token)})")

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print("\n--- repo files: Qwerty0193/genimage-processed ---")
    files = api.list_repo_files("Qwerty0193/genimage-processed", repo_type="dataset")
    for f in files:
        print(f"  {f}")

    print("\n--- repo info: Qwerty0193/genimage-processed ---")
    info = api.dataset_info("Qwerty0193/genimage-processed")
    print(f"  description preview: {(info.description or '')[:400]}")
    if info.card_data:
        cd = info.card_data.to_dict() if hasattr(info.card_data, "to_dict") else dict(info.card_data)
        for k, v in cd.items():
            sv = str(v)
            if len(sv) > 400:
                sv = sv[:400] + "..."
            print(f"  card.{k}: {sv}")


if __name__ == "__main__":
    main()

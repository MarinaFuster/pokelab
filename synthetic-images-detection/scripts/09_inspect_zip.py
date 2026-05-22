"""Print sample zip member names so we know how to address them."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download

from pipeline import load_hf_token


def main() -> None:
    token = load_hf_token()
    zip_path = hf_hub_download(
        repo_id="Qwerty0193/genimage-processed",
        filename="processed_images.zip",
        repo_type="dataset",
        token=token,
    )
    print(f"zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        print(f"total entries: {len(names)}")
        print("first 15 entries:")
        for n in names[:15]:
            print(f"  {n!r}")
        print("\nentries containing 'BigGAN' (first 10):")
        bg = [n for n in names if "BigGAN" in n]
        for n in bg[:10]:
            print(f"  {n!r}")
        print(f"  ... total BigGAN entries: {len(bg)}")
        print("\nentries ending in '158_biggan_00026.png':")
        match = [n for n in names if n.endswith("158_biggan_00026.png")]
        print(f"  {match}")


if __name__ == "__main__":
    main()

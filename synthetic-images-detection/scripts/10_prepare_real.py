"""Prepare real images from ILSVRC/imagenet-1k validation split.

Streams the validation split (avoids the giant full download), filters to
the class indices in our 5 super-cats, samples up to 500 per super-cat,
saves to cache/real/<super_cat>/<idx>.jpg and writes
cache/real_manifest.csv.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from datasets import load_dataset

from pipeline import (
    CACHE_DIR,
    SUPER_CATEGORIES,
    all_super_indices,
    imagenet_class_names,
    load_hf_token,
    super_category_for_idx,
)

PER_CELL_CAP = 500
RNG_SEED = 1


def main() -> None:
    print("[0/3] loading env + names ...", flush=True)
    token = load_hf_token()
    names = imagenet_class_names()
    keep_indices = all_super_indices()
    print(
        f"  keeping {len(keep_indices)} ImageNet class ids across "
        f"{len(SUPER_CATEGORIES)} super-cats",
        flush=True,
    )

    print("[1/3] streaming ILSVRC/imagenet-1k validation ...", flush=True)
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True,
        token=token,
    )

    out_root = CACHE_DIR / "real"
    out_root.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {cat: 0 for cat in SUPER_CATEGORIES}
    caps_full = {cat: False for cat in SUPER_CATEGORIES}
    manifest_rows: list[dict] = []

    print("[2/3] filtering + saving up to 500 per super-cat ...", flush=True)
    seen = 0
    for example in ds:
        seen += 1
        label = int(example["label"])
        if label not in keep_indices:
            continue
        cat = super_category_for_idx(label)
        if cat is None or caps_full[cat]:
            continue
        if counters[cat] >= PER_CELL_CAP:
            caps_full[cat] = True
            if all(caps_full.values()):
                break
            continue

        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        target_dir = out_root / cat
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{label:03d}_{counters[cat]:04d}.jpg"
        if not out_path.is_file():
            img.save(out_path, format="JPEG", quality=96)
        manifest_rows.append(
            {
                "source": "real",
                "super_cat": cat,
                "class_idx": label,
                "class_name": names[label],
                "local_path": str(out_path.relative_to(CACHE_DIR)),
            }
        )
        counters[cat] += 1
        if seen % 1000 == 0:
            done = ", ".join(f"{c}={counters[c]}" for c in SUPER_CATEGORIES)
            print(f"  scanned={seen}, {done}", flush=True)
        if all(caps_full.values()):
            break

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = CACHE_DIR / "real_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\n[3/3] wrote {manifest_path} ({len(manifest)} rows)", flush=True)
    print(
        manifest.groupby(["source", "super_cat"]).size().unstack(fill_value=0).to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()

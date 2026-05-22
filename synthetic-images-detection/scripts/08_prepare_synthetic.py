"""Prepare synthetic image samples (BigGAN-ai, SDv1.5-ai).

Downloads `processed_images.zip` once, then extracts only the files we
need into `cache/synthetic/<source>/<super_cat>/<filename>` and writes
the manifest to `cache/synthetic_manifest.csv`.

Sampling rule (matches the execution-plan adjustment for the smaller
genimage-processed subset): up to 500 images per (source, super_cat);
if fewer are available, take all.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from pipeline import (
    CACHE_DIR,
    SUPER_CATEGORIES,
    extract_ai_class_idx,
    extract_synset,
    imagenet_class_names,
    imagenet_synset_to_idx,
    load_hf_token,
    super_category_for_idx,
)

PER_CELL_CAP = 500
RNG_SEED = 0


def main() -> None:
    print("[0/4] loading env + ImageNet class maps ...", flush=True)
    token = load_hf_token()
    names = imagenet_class_names()
    syn_to_idx = imagenet_synset_to_idx()
    print(f"  ok ({len(names)} class names, {len(syn_to_idx)} synset->idx mappings)", flush=True)

    print("[1/4] downloading metadata.parquet ...", flush=True)
    meta_path = hf_hub_download(
        repo_id="Qwerty0193/genimage-processed",
        filename="metadata.parquet",
        repo_type="dataset",
        token=token,
    )
    df = pd.read_parquet(meta_path)
    df["ai_class_idx"] = df["file_path"].map(extract_ai_class_idx)
    df["synset"] = df["file_path"].map(extract_synset)
    df["nature_class_idx"] = df["synset"].map(lambda s: syn_to_idx.get(s) if s else None)
    df["class_idx"] = df["ai_class_idx"].fillna(df["nature_class_idx"])
    df["super_cat"] = df["class_idx"].map(
        lambda x: super_category_for_idx(int(x)) if pd.notna(x) else None
    )

    # Sample BigGAN-ai and SDv1.5-ai rows by super_cat
    selected_frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(RNG_SEED)
    for source_label, gen in [("biggan", "BigGAN"), ("sdv1_5", "Stable Diffusion V1.5")]:
        for cat in SUPER_CATEGORIES:
            sub = df[
                (df["generator"] == gen)
                & (df["label_name"] == "ai")
                & (df["super_cat"] == cat)
            ]
            n = min(len(sub), PER_CELL_CAP)
            if n == 0:
                print(f"  WARN: 0 rows for {source_label}/{cat}")
                continue
            picked = sub.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))
            picked = picked.assign(source=source_label, super_cat=cat)
            selected_frames.append(picked)
    sel = pd.concat(selected_frames, ignore_index=True)
    print(f"[2/4] selected {len(sel)} synthetic rows")
    print(sel.groupby(["source", "super_cat"]).size().unstack(fill_value=0).to_string())

    print("\n[3/4] downloading processed_images.zip (this is large; cached after first run)...")
    zip_path = hf_hub_download(
        repo_id="Qwerty0193/genimage-processed",
        filename="processed_images.zip",
        repo_type="dataset",
        token=token,
    )
    print(f"  zip cached at {zip_path}")

    out_root = CACHE_DIR / "synthetic"
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n[4/4] extracting selected files from zip ...", flush=True)
    manifest_rows = []
    n_missing = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        zip_names_lower = {n.lower(): n for n in z.namelist()}
        for i, row in enumerate(sel.itertuples(index=False)):
            # metadata 'file_path' uses ORIGINAL kaggle split (train|val);
            # zip uses NEW 70/15/15 split column 'split' (train|val|test).
            fp = row.file_path
            parts = fp.replace("\\", "/").split("/")
            try:
                anchor = parts.index("genimage_subset")
            except ValueError:
                anchor = -1
            if anchor < 0 or anchor + 4 >= len(parts):
                n_missing += 1
                if n_missing <= 5:
                    print(f"  MISSING anchor: {fp}", flush=True)
                continue
            gen = parts[anchor + 1]
            label = parts[anchor + 3]
            filename = parts[anchor + 4]
            stem = Path(filename).stem
            zip_member_guess = f"{row.split}/{label}/{gen}/{stem}.jpg"
            zip_member = zip_names_lower.get(zip_member_guess.lower())
            if zip_member is None:
                n_missing += 1
                if n_missing <= 5:
                    print(
                        f"  MISSING in zip: tried {zip_member_guess!r} for {fp!r}",
                        flush=True,
                    )
                continue

            target_dir = out_root / row.source / row.super_cat
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / Path(zip_member).name
            if not target_path.is_file():
                with z.open(zip_member) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            manifest_rows.append(
                {
                    "source": row.source,
                    "super_cat": row.super_cat,
                    "class_idx": int(row.class_idx),
                    "class_name": names[int(row.class_idx)],
                    "local_path": str(target_path.relative_to(CACHE_DIR)),
                    "zip_member": zip_member,
                }
            )
            if (i + 1) % 200 == 0:
                print(f"  extracted {i + 1}/{len(sel)}", flush=True)
    if n_missing:
        print(f"  total missing: {n_missing}/{len(sel)}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = CACHE_DIR / "synthetic_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\n[done] wrote {manifest_path} ({len(manifest)} rows)", flush=True)
    if len(manifest):
        print(
            manifest.groupby(["source", "super_cat"]).size().unstack(fill_value=0).to_string(),
            flush=True,
        )


if __name__ == "__main__":
    main()

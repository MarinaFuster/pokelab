"""Rebuild cache/real_manifest.csv from images already on disk.

Real-image filenames are like '<class_idx_3digit>_<sample_4digit>.jpg'
inside cache/real/<super_cat>/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pipeline import CACHE_DIR, SUPER_CATEGORIES, imagenet_class_names

FNAME_RE = re.compile(r"^(\d{3})_(\d{4})\.jpg$", re.IGNORECASE)


def main() -> None:
    names = imagenet_class_names()
    rows: list[dict] = []
    real_root = CACHE_DIR / "real"
    for cat in SUPER_CATEGORIES:
        cat_dir = real_root / cat
        if not cat_dir.is_dir():
            print(f"  WARN: {cat_dir} missing")
            continue
        files = sorted(cat_dir.glob("*.jpg"))
        for p in files:
            m = FNAME_RE.match(p.name)
            if not m:
                print(f"  WARN: skipping {p}")
                continue
            cls_idx = int(m.group(1))
            rows.append(
                {
                    "source": "real",
                    "super_cat": cat,
                    "class_idx": cls_idx,
                    "class_name": names[cls_idx],
                    "local_path": str(p.relative_to(CACHE_DIR)),
                }
            )

    df = pd.DataFrame(rows)
    out_path = CACHE_DIR / "real_manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")
    print(df.groupby(["source", "super_cat"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()

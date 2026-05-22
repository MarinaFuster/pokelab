"""Print names of all class indices in the 'vehicle' super-cat to audit them."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import SUPER_CATEGORIES, imagenet_class_names


def main() -> None:
    names = imagenet_class_names()
    for cat, ids in SUPER_CATEGORIES.items():
        print(f"\n== {cat} ({len(ids)} ids) ==")
        for i in sorted(ids):
            print(f"  {i:3d}  {names[i]}")


if __name__ == "__main__":
    main()

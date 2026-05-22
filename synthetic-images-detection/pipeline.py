"""Shared utilities for the synthetic-images-detection experiment.

Provides:
- env loading and HF_TOKEN access
- super-category definitions (5 broad ImageNet super-cats)
- class-index extraction from genimage AI filenames
- synset extraction from genimage nature filenames
- ImageNet idx <-> synset <-> name maps
"""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_ROOT / "results"
CACHE_DIR = EXPERIMENT_ROOT / "cache"

SYNSET_RE = re.compile(r"(n\d{8})_")
AI_CLASS_RE = re.compile(r"/(\d{3})_(?:biggan|sdv5|adm|midjourney)_\d+\.png", re.IGNORECASE)

IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LOC_SYNSET_MAPPING_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)


SUPER_CATEGORIES: dict[str, set[int]] = {
    # Choices verified manually against the imagenet_classes.txt list:
    # animal — dog breeds occupy a tight contiguous range
    "dog": set(range(151, 269)),
    # animal — birds across several ranges
    "bird": set(range(7, 25)) | set(range(80, 101)) | set(range(127, 147)),
    # man-made — wheeled / water / air vehicles (cars, trucks, ships, planes)
    # 408 = "amphibian" in ImageNet is amphibious assault vehicle, not the animal
    "vehicle": {
        404, 407, 408, 436, 444, 468, 472, 484, 510, 511, 554, 555,
        569, 573, 575, 576, 609, 625, 627, 628, 656, 661, 670, 671, 675,
        694, 705, 717, 734, 751, 779, 803, 814, 817, 829, 847, 864, 866,
        867, 870, 880, 895,
    },
    # food / plant (range covers ImageNet's prepared food + produce block)
    "food": set(range(924, 970)),
    # building / structure (manually curated for semantic coherence)
    "structure": {
        410, 425, 442, 449, 483, 497, 525, 538, 562, 624, 649, 663,
        668, 698, 718, 727, 762, 819, 825, 832,
    },
}


def load_hf_token() -> str:
    for env_path in (EXPERIMENT_ROOT / ".env", PROJECT_ROOT / ".env"):
        if env_path.is_file():
            load_dotenv(env_path, override=False)
    token = os.environ.get("HF_TOKEN", "").strip().strip('"').strip("'")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it to .env at project root or experiment folder."
        )
    os.environ["HF_TOKEN"] = token
    return token


def imagenet_class_names() -> list[str]:
    """idx -> human-readable class name (length 1000)."""
    cache = RESULTS_DIR / "imagenet_classes.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file():
        with urllib.request.urlopen(IMAGENET_CLASSES_URL, timeout=30) as r:
            cache.write_bytes(r.read())
    names = cache.read_text(encoding="utf-8").splitlines()
    assert len(names) == 1000
    return names


def imagenet_idx_to_synset() -> dict[int, str]:
    """idx -> synset id (e.g. 0 -> n01440764)."""
    cache = RESULTS_DIR / "imagenet_class_index.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file():
        with urllib.request.urlopen(LOC_SYNSET_MAPPING_URL, timeout=30) as r:
            cache.write_bytes(r.read())
    import json

    raw = json.loads(cache.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = v[0]
    assert len(out) == 1000
    return out


def imagenet_synset_to_idx() -> dict[str, int]:
    return {s: i for i, s in imagenet_idx_to_synset().items()}


def extract_ai_class_idx(file_path: str) -> int | None:
    m = AI_CLASS_RE.search(file_path)
    return int(m.group(1)) if m else None


def extract_synset(file_path: str) -> str | None:
    m = SYNSET_RE.search(file_path)
    return m.group(1) if m else None


def super_category_for_idx(idx: int) -> str | None:
    for name, ids in SUPER_CATEGORIES.items():
        if idx in ids:
            return name
    return None


def all_super_indices() -> set[int]:
    out: set[int] = set()
    for ids in SUPER_CATEGORIES.values():
        out |= ids
    return out


def category_summary(names: Iterable[str] | None = None) -> str:
    if names is None:
        names = imagenet_class_names()
    names = list(names)
    lines = []
    for cat, ids in SUPER_CATEGORIES.items():
        lines.append(f"== {cat} ({len(ids)} class ids) ==")
        for i in sorted(ids):
            lines.append(f"  {i:3d}  {names[i]}")
    return "\n".join(lines)

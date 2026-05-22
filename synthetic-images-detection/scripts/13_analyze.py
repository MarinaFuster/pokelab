"""Run UMAP + distance / silhouette metrics on the three feature spaces.

Produces:
- results/umap_clip.png
- results/umap_fft.png
- results/umap_srm.png
- results/metrics.csv (one row per feature space x measurement)

The five measurements per feature space, matching the execution plan:
1. Mean intra-class distance (same source AND same category)
2. Mean inter-source: BigGAN vs. SDv1.5 (same category)
3. Mean inter-source: real vs. fake (same category)
4. Mean inter-category: real vs. real (different categories)
5. Silhouette score for 3-class label (real / GAN / diffusion)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

from pipeline import CACHE_DIR, RESULTS_DIR, SUPER_CATEGORIES

SOURCE_ORDER = ["real", "biggan", "sdv1_5"]
SUPER_CAT_ORDER = list(SUPER_CATEGORIES.keys())
SOURCE_COLORS = {"real": "#1f77b4", "biggan": "#d62728", "sdv1_5": "#2ca02c"}
SUPER_CAT_MARKERS = {
    "dog": "o",
    "bird": "s",
    "vehicle": "^",
    "food": "D",
    "structure": "P",
}


def load_features() -> dict:
    arr = np.load(CACHE_DIR / "features.npz", allow_pickle=True)
    return {
        "clip": arr["clip"].astype(np.float32),
        "fft": arr["fft"].astype(np.float32),
        "srm": arr["srm"].astype(np.float32),
        "source": np.array(arr["source"]),
        "super_cat": np.array(arr["super_cat"]),
        "class_idx": np.array(arr["class_idx"]).astype(int),
    }


def run_umap(feat: np.ndarray, metric: str) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        metric=metric,
        random_state=0,
        verbose=False,
    )
    return reducer.fit_transform(feat)


def plot_umap(emb: np.ndarray, sources: np.ndarray, cats: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for src in SOURCE_ORDER:
        for cat in SUPER_CAT_ORDER:
            mask = (sources == src) & (cats == cat)
            if not mask.any():
                continue
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=SOURCE_COLORS[src],
                marker=SUPER_CAT_MARKERS[cat],
                s=20,
                alpha=0.55,
                edgecolors="none",
                label=f"{src} / {cat}",
            )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    src_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=SOURCE_COLORS[s],
                   markersize=10, label=s, linestyle="")
        for s in SOURCE_ORDER
    ]
    cat_handles = [
        plt.Line2D([0], [0], marker=SUPER_CAT_MARKERS[c], color="black", markersize=10,
                   label=c, linestyle="", markerfacecolor="lightgrey")
        for c in SUPER_CAT_ORDER
    ]
    leg1 = ax.legend(handles=src_handles, title="source", loc="upper left", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=cat_handles, title="super-cat", loc="lower left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _mean_offdiag(block: np.ndarray) -> float:
    """Mean of off-diagonal entries of a square distance block."""
    n = block.shape[0]
    if n <= 1:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(block[mask].mean())


def compute_metrics(feat: np.ndarray, sources: np.ndarray, cats: np.ndarray) -> dict[str, float]:
    """Return the five measurements (all mean L2 distances + silhouette)."""
    D = cdist(feat, feat, metric="euclidean")
    n = feat.shape[0]

    # 1. mean intra-class (same source AND same category)
    intra_vals: list[float] = []
    for src in SOURCE_ORDER:
        for cat in SUPER_CAT_ORDER:
            mask = (sources == src) & (cats == cat)
            idx = np.where(mask)[0]
            if len(idx) >= 2:
                block = D[np.ix_(idx, idx)]
                intra_vals.append(_mean_offdiag(block))
    intra = float(np.mean(intra_vals)) if intra_vals else float("nan")

    # 2. inter-source BigGAN vs SDv1.5 (same category)
    bg_sd_vals: list[float] = []
    for cat in SUPER_CAT_ORDER:
        bg_idx = np.where((sources == "biggan") & (cats == cat))[0]
        sd_idx = np.where((sources == "sdv1_5") & (cats == cat))[0]
        if len(bg_idx) and len(sd_idx):
            bg_sd_vals.append(float(D[np.ix_(bg_idx, sd_idx)].mean()))
    bg_sd = float(np.mean(bg_sd_vals)) if bg_sd_vals else float("nan")

    # 3. inter-source real vs fake (same category); fake = biggan + sdv1_5
    real_fake_vals: list[float] = []
    for cat in SUPER_CAT_ORDER:
        real_idx = np.where((sources == "real") & (cats == cat))[0]
        fake_idx = np.where(((sources == "biggan") | (sources == "sdv1_5")) & (cats == cat))[0]
        if len(real_idx) and len(fake_idx):
            real_fake_vals.append(float(D[np.ix_(real_idx, fake_idx)].mean()))
    real_fake = float(np.mean(real_fake_vals)) if real_fake_vals else float("nan")

    # 4. inter-category among real (different categories)
    inter_cat_vals: list[float] = []
    for i, cat_a in enumerate(SUPER_CAT_ORDER):
        a_idx = np.where((sources == "real") & (cats == cat_a))[0]
        for cat_b in SUPER_CAT_ORDER[i + 1 :]:
            b_idx = np.where((sources == "real") & (cats == cat_b))[0]
            if len(a_idx) and len(b_idx):
                inter_cat_vals.append(float(D[np.ix_(a_idx, b_idx)].mean()))
    inter_cat = float(np.mean(inter_cat_vals)) if inter_cat_vals else float("nan")

    # 5. silhouette for 3-class label (real / GAN / diffusion), ignoring category
    three_class = np.where(
        sources == "real", "real",
        np.where(sources == "biggan", "GAN", "diffusion"),
    )
    sil = float(
        silhouette_score(feat, three_class, metric="euclidean", sample_size=min(2000, n), random_state=0)
    )
    # Also compute pairwise silhouettes for individual contrasts
    sil_real_vs_fake = float(
        silhouette_score(
            feat,
            np.where(sources == "real", "real", "fake"),
            metric="euclidean",
            sample_size=min(2000, n),
            random_state=0,
        )
    )
    fake_mask = sources != "real"
    if fake_mask.sum() > 10:
        sil_gan_vs_diff = float(
            silhouette_score(
                feat[fake_mask],
                np.where(sources[fake_mask] == "biggan", "GAN", "diffusion"),
                metric="euclidean",
                sample_size=min(2000, fake_mask.sum()),
                random_state=0,
            )
        )
    else:
        sil_gan_vs_diff = float("nan")

    return {
        "intra (same source & cat)": intra,
        "inter biggan-vs-sdv15 (same cat)": bg_sd,
        "inter real-vs-fake (same cat)": real_fake,
        "inter real-cat A vs real-cat B": inter_cat,
        "silhouette 3-class (real/GAN/diff)": sil,
        "silhouette real vs fake": sil_real_vs_fake,
        "silhouette GAN vs diff (fakes only)": sil_gan_vs_diff,
    }


def main() -> None:
    print("[load] features.npz", flush=True)
    data = load_features()
    sources = data["source"]
    cats = data["super_cat"]
    n = len(sources)
    print(f"  N = {n}")
    print(
        pd.crosstab(pd.Series(sources, name="source"), pd.Series(cats, name="super_cat")).to_string()
    )

    spaces = [
        ("clip", "cosine", "Baseline 1 — Semantic (CLIP ViT-B/32)"),
        ("fft", "euclidean", "Experiment 2a — FFT log-magnitude (upper-left 64x64)"),
        ("srm", "euclidean", "Experiment 2b — SRM high-pass residual (PCA-256)"),
    ]

    metric_rows = []
    for key, umap_metric, title in spaces:
        print(f"\n[{key}] features: {data[key].shape}, umap metric={umap_metric}")
        emb = run_umap(data[key], metric=umap_metric)
        out_path = RESULTS_DIR / f"umap_{key}.png"
        plot_umap(emb, sources, cats, title, out_path)
        print(f"  wrote {out_path}")

        m = compute_metrics(data[key], sources, cats)
        for k, v in m.items():
            metric_rows.append({"feature_space": key, "measurement": k, "value": v})
        print("  metrics:")
        for k, v in m.items():
            print(f"    {k:42s} = {v: .6f}")

    df = pd.DataFrame(metric_rows)
    out_csv = RESULTS_DIR / "metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {out_csv}")

    pivot = df.pivot(index="measurement", columns="feature_space", values="value")
    pivot = pivot[["clip", "fft", "srm"]]
    print("\n=== summary table ===")
    print(pivot.to_string(float_format=lambda x: f"{x: .4f}"))
    pivot.to_csv(RESULTS_DIR / "metrics_pivot.csv")


if __name__ == "__main__":
    main()

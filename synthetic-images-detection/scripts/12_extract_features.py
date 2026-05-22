"""Extract three feature representations from the prepared image set.

For each image (real / BigGAN / SDv1.5, across 5 super-cats):

- CLIP ViT-B/32 image embeddings (semantic baseline, 512-d)
- FFT log-magnitude: fftshift-centered spectrum, upper-left 64x64 corner
  (captures mid-to-high frequencies where GAN/diffusion artifacts live, 4096-d)
- SRM high-pass residuals -> PCA to 256-d (artifact / noise space)

All features and labels are saved to cache/features.npz so the
analysis step is cheap to re-run.

Pass --only-fft to recompute only the FFT features and patch the existing
features.npz in-place (CLIP and SRM are preserved as-is).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

from pipeline import CACHE_DIR

IMG_SIZE = 224
FFT_QUAD = 64
SRM_PCA_DIM = 256
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_BATCH = 64

SRM_KERNEL = np.array(
    [
        [-1.0, 2.0, -1.0],
        [2.0, -4.0, 2.0],
        [-1.0, 2.0, -1.0],
    ]
) / 4.0


def load_manifests() -> pd.DataFrame:
    synth = pd.read_csv(CACHE_DIR / "synthetic_manifest.csv")
    real = pd.read_csv(CACHE_DIR / "real_manifest.csv")
    df = pd.concat([real, synth], ignore_index=True)
    df["abs_path"] = df["local_path"].map(lambda p: str(CACHE_DIR / p))
    return df


def load_images(df: pd.DataFrame) -> np.ndarray:
    """Return uint8 array (N, 224, 224, 3)."""
    arr = np.empty((len(df), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(df["abs_path"]):
        img = Image.open(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr[i] = np.asarray(img, dtype=np.uint8)
        if (i + 1) % 500 == 0:
            print(f"  loaded {i + 1}/{len(df)}", flush=True)
    return arr


def extract_clip(images: np.ndarray) -> np.ndarray:
    print(f"[clip] loading {CLIP_MODEL_ID} ...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    model.eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    embs: list[np.ndarray] = []
    n = len(images)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, n, CLIP_BATCH):
            stop = min(start + CLIP_BATCH, n)
            batch_pil = [Image.fromarray(images[i]) for i in range(start, stop)]
            inputs = proc(images=batch_pil, return_tensors="pt")
            pixel = inputs["pixel_values"].to(device)
            # transformers>=5 changed get_image_features to return the vision
            # tower output; we apply the visual projection ourselves to get the
            # standard 512-d projected CLIP image embedding.
            vision_out = model.vision_model(pixel_values=pixel)
            pooled = vision_out.pooler_output  # (B, vision_dim=768)
            feats = model.visual_projection(pooled)  # (B, 512)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            embs.append(feats.cpu().numpy().astype(np.float32))
            if (stop) % (CLIP_BATCH * 4) == 0 or stop == n:
                elapsed = time.time() - t0
                rate = stop / max(elapsed, 1e-6)
                eta = (n - stop) / max(rate, 1e-6)
                print(
                    f"  clip {stop}/{n}  "
                    f"({rate:.1f} img/s, ETA {eta:.0f}s)",
                    flush=True,
                )
    out = np.concatenate(embs, axis=0)
    print(f"[clip] done -> {out.shape} (l2-normalised)", flush=True)
    return out


def extract_fft(images: np.ndarray) -> np.ndarray:
    """Per-image: grayscale -> 2D FFT -> fftshift -> log mag -> upper-left 64x64 -> flatten.

    fftshift recentres the DC component to (H/2, W/2), so the upper-left 64x64
    corner of the shifted spectrum corresponds to the mid-to-high frequency
    region — where GAN checkerboard and diffusion pipeline artefacts manifest.
    """
    print(f"[fft] computing on {len(images)} images ...", flush=True)
    t0 = time.time()
    # luminance-Y as a single float32 array (N, H, W)
    rgb = images.astype(np.float32) / 255.0
    gray = (
        0.2989 * rgb[..., 0]
        + 0.5870 * rgb[..., 1]
        + 0.1140 * rgb[..., 2]
    )
    n = gray.shape[0]
    out = np.empty((n, FFT_QUAD * FFT_QUAD), dtype=np.float32)
    # Compute in chunks to keep peak memory in check
    chunk = 256
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        f = np.fft.fftshift(np.fft.fft2(gray[start:stop]))
        mag = np.log1p(np.abs(f))
        out[start:stop] = mag[:, :FFT_QUAD, :FFT_QUAD].reshape(stop - start, -1)
    print(
        f"[fft] done -> {out.shape} (took {time.time() - t0:.1f}s)",
        flush=True,
    )
    return out


def extract_srm(images: np.ndarray, pca_dim: int = SRM_PCA_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Per-image: 3x3 SRM high-pass per channel -> flatten -> PCA -> pca_dim.

    Returns (pca_features, raw_l2_norms) where raw_l2_norms is the per-image
    L2 norm of the (un-PCA) residual map for sanity-checking.
    """
    print(f"[srm] convolving high-pass on {len(images)} images ...", flush=True)
    t0 = time.time()
    n = len(images)
    rgb = images.astype(np.float32) / 255.0
    feats = np.empty((n, 3 * IMG_SIZE * IMG_SIZE), dtype=np.float32)
    norms = np.empty(n, dtype=np.float32)
    for i in range(n):
        residual = np.empty((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        for c in range(3):
            residual[..., c] = convolve2d(rgb[i, ..., c], SRM_KERNEL, mode="same", boundary="symm")
        flat = residual.reshape(-1)
        feats[i] = flat
        norms[i] = float(np.linalg.norm(flat))
        if (i + 1) % 500 == 0:
            print(
                f"  srm {i + 1}/{n}  (elapsed {time.time() - t0:.1f}s)",
                flush=True,
            )
    print(
        f"[srm] raw residual -> {feats.shape}; running PCA to {pca_dim} dims...",
        flush=True,
    )
    pca = PCA(n_components=pca_dim, random_state=0)
    pca_feats = pca.fit_transform(feats).astype(np.float32)
    print(
        f"[srm] pca done. explained variance ratio sum = "
        f"{pca.explained_variance_ratio_.sum():.4f}; "
        f"output {pca_feats.shape} (took {time.time() - t0:.1f}s)",
        flush=True,
    )
    return pca_feats, norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-fft",
        action="store_true",
        help="Recompute only FFT features and patch the existing features.npz in-place.",
    )
    args = parser.parse_args()

    out_path = CACHE_DIR / "features.npz"

    if args.only_fft:
        if not out_path.is_file():
            print(f"[error] {out_path} not found. Run without --only-fft first.")
            return
        print("[only-fft] loading existing features.npz ...", flush=True)
        existing = np.load(out_path, allow_pickle=True)

        print("[only-fft] loading manifests ...", flush=True)
        df = load_manifests()

        print(f"[only-fft] loading {len(df)} images into memory ...", flush=True)
        t0 = time.time()
        images = load_images(df)
        print(f"  done in {time.time() - t0:.1f}s", flush=True)

        print("[only-fft] recomputing FFT features ...", flush=True)
        fft_feats = extract_fft(images)

        np.savez_compressed(
            out_path,
            clip=existing["clip"],
            fft=fft_feats,
            srm=existing["srm"],
            srm_norms=existing["srm_norms"],
            source=existing["source"],
            super_cat=existing["super_cat"],
            class_idx=existing["class_idx"],
            local_path=existing["local_path"],
        )
        print(f"[only-fft] patched {out_path}", flush=True)
        return

    if out_path.is_file():
        print(f"[skip] {out_path} already exists; delete to regenerate.")
        return

    print("[1/5] loading manifests ...", flush=True)
    df = load_manifests()
    print(
        df.groupby(["source", "super_cat"]).size().unstack(fill_value=0).to_string(),
        flush=True,
    )

    print(f"\n[2/5] loading {len(df)} images into memory ...", flush=True)
    t0 = time.time()
    images = load_images(df)
    print(
        f"  done. arr.shape={images.shape}, "
        f"mem={images.nbytes / 1e6:.1f} MB, "
        f"took {time.time() - t0:.1f}s",
        flush=True,
    )

    print("\n[3/5] CLIP features ...", flush=True)
    clip_feats = extract_clip(images)

    print("\n[4/5] FFT features ...", flush=True)
    fft_feats = extract_fft(images)

    print("\n[5/5] SRM features (with PCA) ...", flush=True)
    srm_feats, srm_norms = extract_srm(images)

    sources = df["source"].to_numpy()
    super_cats = df["super_cat"].to_numpy()
    class_idx = df["class_idx"].to_numpy()
    paths = df["local_path"].to_numpy()

    np.savez_compressed(
        out_path,
        clip=clip_feats,
        fft=fft_feats,
        srm=srm_feats,
        srm_norms=srm_norms,
        source=sources,
        super_cat=super_cats,
        class_idx=class_idx,
        local_path=paths,
    )
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

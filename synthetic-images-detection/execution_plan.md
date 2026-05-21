# Execution Plan

## Context
This experiment tests whether GAN-generated and diffusion-generated images share a common artifact signature that is consistently separable from real camera-pipeline images — i.e., whether "synthetic" is a valid single detectable class. Three feature spaces are compared: semantic embeddings (CLIP, as a baseline confirming distributional overlap), FFT magnitude spectra (frequency-domain pipeline traces), and SRM noise residuals (spatial noise pipeline traces). The key signal is whether the fake-GAN vs. fake-diffusion distance is small (similar to same-pipeline real-real distance) and whether real vs. fake distance is large, across artifact-sensitive feature spaces.

## Models / Datasets
- **Synthetic — GAN**: BigGAN images from `Qwerty0193/genimage-processed` (split: `biggan`) — BigGAN is a canonical, widely-adopted class-conditional GAN trained on ImageNet
- **Synthetic — Diffusion**: Stable Diffusion v1.5 images from `Qwerty0193/genimage-processed` (split: `sdv1_5`) — SD v1.5 is the most widely deployed open-source diffusion model
- **Real**: `ILSVRC/imagenet-1k` validation split (requires `HF_TOKEN` and license acceptance)
- **Semantic feature extractor**: CLIP ViT-B/32 via `openai/clip-vit-base-patch32` (HuggingFace transformers)

## Prerequisites
- `.env` file in the project root with `HF_TOKEN=<your_token>` — agent must load this at startup and raise a clear error if missing or empty
- ImageNet license accepted on HuggingFace (see `HUMAN_SETUP.md`)

## Steps

1. **Create and activate a virtual environment**:
   - Create `.venv` inside `synthetic-images-detection/` if it does not already exist: `python -m venv .venv`
   - All subsequent `pip install` and `python` calls must use the interpreter at `.venv/Scripts/python` (Windows) or `.venv/bin/python` (Unix)
   - Install all dependencies into that environment: `pip install python-dotenv datasets transformers torch torchvision umap-learn scikit-learn numpy scipy matplotlib seaborn Pillow`

2. **Verify environment**: Load `.env` (from `synthetic-images-detection/.env` or project root), assert `HF_TOKEN` is set and non-empty; abort with a descriptive message if not.

2. **Load synthetic dataset**: Download `Qwerty0193/genimage-processed`; extract the `biggan` and `sdv1_5` splits. Inspect available columns — confirm there is a label/class column aligned with ImageNet class IDs or names.

3. **Select categories**: From the ImageNet classes available in both synthetic splits, pick N=5 categories that are semantically diverse — at minimum one animal, one vehicle, one man-made object, one food/plant, one scene/structure (e.g., "golden retriever", "sports car", "church", "banana", "bookshelf" — adjust to whatever classes are present in the processed dataset).

4. **Load real dataset**: Load `ILSVRC/imagenet-1k` validation split using `HF_TOKEN`; filter to the same N=5 categories selected in step 3.

5. **Sample**: For each of the 3 sources (BigGAN, SD v1.5, real) × 5 categories, sample up to 500 images. If a source has fewer than 500 for a given category, use all available and note the count.

6. **Preprocess images**: Resize all images to 224×224 RGB. Keep two versions of each image: (a) normalized tensor for CLIP, (b) raw uint8 numpy array for FFT and SRM.

7. **Baseline 1 — Semantic (CLIP)**:
   - Extract CLIP ViT-B/32 embeddings for all images (batch size 64)
   - Run UMAP (n_components=2, n_neighbors=30, metric='cosine')
   - Plot: color by source (real / BigGAN / SD v1.5), marker by category
   - Save plot to `results/umap_clip.png`
   - Expected result: fakes cluster near real images of the same content category → confirms distributional overlap

8. **Experiment 2a — Frequency (FFT)**:
   - For each image: convert to grayscale, compute 2D FFT, take log magnitude, flatten the upper-left 64×64 quadrant (low/mid frequencies) as the feature vector
   - Run UMAP (n_components=2, n_neighbors=30, metric='euclidean')
   - Plot: same color/marker scheme; save to `results/umap_fft.png`
   - Compute pairwise L2 distance matrix; extract the three comparison distances (see Metrics)

9. **Experiment 2b — Noise residual (SRM)**:
   - For each image: apply a 3×3 SRM high-pass filter (kernel: `[[-1,2,-1],[2,-4,2],[-1,2,-1]] / 4`) to each channel; flatten the residual as the feature vector (optionally PCA to 256 dims before UMAP for speed)
   - Run UMAP (n_components=2, n_neighbors=30, metric='euclidean')
   - Plot: same scheme; save to `results/umap_srm.png`
   - Compute pairwise distances same as 2a

10. **Metrics**: For each feature space (CLIP, FFT, SRM), compute and print a table with:
    - Mean intra-class distance (same source, same category)
    - Mean inter-source distance: BigGAN vs. SD v1.5 (same category)
    - Mean inter-source distance: real vs. fake (same category)
    - Mean inter-category distance: real cat A vs. real cat B (same-pipeline reference)
    - Silhouette score for a 3-class label (real / GAN / diffusion), ignoring category

11. **Save results**: Write the metrics table to `results/metrics.csv` and print a summary to stdout.

12. **Notebook**: Wrap all steps in `experiment.ipynb` with one cell per step and markdown headers. Each UMAP plot and the metrics table should render inline.

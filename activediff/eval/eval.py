
#!/usr/bin/env python3
"""Concatenate images and FOM from all iteration folders, then run eval_single_file_standalone."""

import os
import sys
import glob
import yaml
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import activediff
from activediff.utils import compute_distances, dist_select, binarisation, filter_similar_samples

from eval_single_file_standalone import eval_single_file, CompareToTrainClosestImage

def normalize_binarize(images: torch.tensor, threshold: float = 0.5) -> torch.Tensor:
    """Binarize images using a simple threshold."""
    images = (images - images.min()) / (images.max() - images.min() + 1e-8)
    images = (images > threshold).float()
    return images

BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
distance_select = True

# --- Font scale config (1.0 = matplotlib defaults) ---
FONT_SCALE = 2.0

_base = plt.rcParams["font.size"]
plt.rcParams.update({
    "font.size":         _base * FONT_SCALE,
    "axes.titlesize":    _base * FONT_SCALE,
    "axes.labelsize":    _base * FONT_SCALE,
    "xtick.labelsize":   _base * FONT_SCALE,
    "ytick.labelsize":   _base * FONT_SCALE,
    "legend.fontsize":   _base * FONT_SCALE,
    "figure.titlesize":  _base * FONT_SCALE,
})

all_images = []
all_foms = []
# Cache loaded data to avoid re-reading from disk later
loaded_data: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

# Find all image files recursively under BASE_DIR, sorted for reproducibility
# Exclude the "all/" aggregation directory
img_files_all = sorted(
    f for f in glob.glob(os.path.join(BASE_DIR, "**", "selected_samples_iter_*.pt"), recursive=True)
    if Path(f).parent.name != "all"
)

for img_file in img_files_all:
    folder = os.path.dirname(img_file)
    # Extract iteration index from filename and save results in BASE_DIR/<i>/
    stem = Path(img_file).stem  # e.g. "selected_samples_iter_0"
    iter_idx = stem.split("_")[-1]
    savepath = Path(BASE_DIR) / iter_idx
    savepath.mkdir(exist_ok=True)
    fom_files = glob.glob(os.path.join(folder, f"selected_samples_fom_iter_{iter_idx}.pt"))
    if not fom_files:
        fom_files = glob.glob(os.path.join(folder, f"selected_fom_scores_iter_{iter_idx}.pt"))
    print(f"files found in {folder}")
    imgs = normalize_binarize(torch.load(img_file)).numpy()
    foms = torch.load(fom_files[0]).numpy() if fom_files else None
    if foms is not None:
        print(f"  FOM file found: {fom_files[0]}")
        assert foms.shape[0] == imgs.shape[0], f"FOM count {foms.shape[0]} does not match image count {imgs.shape[0]} in {folder}"
    eval_single_file(images=imgs, savepath=savepath, fom=foms, force_recompute=True)

    loaded_data[img_file] = (imgs, foms)
    all_images.append(imgs)
    print(f"  iter {iter_idx}: {imgs.shape[0]} images")
    if foms is not None:
        all_foms.append(foms)
    

all_images = np.concatenate(all_images)
all_foms = np.concatenate(all_foms) if all_foms else None

if distance_select:
    init_n_images = all_images.shape[0]
    images = torch.from_numpy(all_images)
    fom = all_foms
    train_set = np.load(Path(activediff.__file__).parent.parent / 'data/imagesnorm.npy')
    train_set = torch.from_numpy(train_set).float()
    # dist select
    distances = compute_distances(images, train_set)
    distance_mask = distances > 10
    print(f"Distance selection: {distance_mask.sum()} images selected out of {len(distance_mask)}")
    images = images[distance_mask]
    distances = distances[distance_mask]
    fom = fom[distance_mask]
    images, fom = filter_similar_samples(images, fom,
                                                  distance_threshold=10)
    print(f"Filtering removed {init_n_images - images.shape[0]} similar samples: {images.shape[0]} images remain")
    all_images = images.numpy()
    all_foms = fom

print(f"Total: {all_images.shape[0]} images")

savepath = Path(os.path.join(BASE_DIR, "all"))
savepath.mkdir(exist_ok=True)

eval_single_file(images=all_images, savepath=savepath, fom=all_foms, force_recompute=True)

# plot_stats
#!/usr/bin/env python3
"""Plot FOM (box plot) and NNDistanceTrainSet across experiment indices."""


fom_data = []
nn_data = []
pca_entropy_data = []
valid_indices = []

for i in sorted(set(
    Path(f).stem.split("_")[-1]
    for f in img_files_all
), key=lambda x: int(x)):
    stats_path = os.path.join(BASE_DIR, str(i), "stats.yaml")
    if not os.path.exists(stats_path):
        continue
    with open(stats_path) as f:
        stats = yaml.safe_load(f)
    # Load raw FOM values — match exact iter index
    fom_files = glob.glob(os.path.join(BASE_DIR, f"selected_samples_fom_iter_{i}.pt"))
    if not fom_files:
        fom_files = glob.glob(os.path.join(BASE_DIR, f"selected_fom_scores_iter_{i}.pt"))
    if not fom_files:
        print(f"  WARNING: no FOM file found for iter {i}, skipping.")
        continue
    fom_vals = torch.load(fom_files[0]).numpy()
    fom_data.append(fom_vals)
    # Load raw NN distances
    nn_path = os.path.join(BASE_DIR, str(i), "nn_distances.npy")
    if os.path.exists(nn_path):
        nn_data.append(np.load(nn_path))
    # Load PCA Projection Per Dimension Entropy
    if "PCAProjPerDimEntropy" in stats:
        pca_entropy_data.append(stats["PCAProjPerDimEntropy"])
    valid_indices.append(i)

x = np.array(valid_indices)
labels = [str(i) for i in valid_indices]
cmap = plt.cm.tab20

# FOM box plot
fig1, ax1 = plt.subplots(figsize=(6, 5))
bp1 = ax1.boxplot(fom_data, patch_artist=True, showfliers=False,
                  medianprops=dict(color='black', linewidth=1.5),
                  boxprops=dict(facecolor=cmap(0), alpha=0.7))
ax1.set_xticklabels(labels)
ax1.set_xlabel("ASG Iteration")
ax1.set_ylabel("FOM")
ax1.set_title("FOM distribution")
fom_path = os.path.join(BASE_DIR, "fom_comparison.pdf")
fig1.savefig(fom_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"Saved to {fom_path}")

# NN Distance box plot
fig2, ax2 = plt.subplots(figsize=(6, 5))
if nn_data:
    bp2 = ax2.boxplot(nn_data, patch_artist=True, showfliers=False,
                      medianprops=dict(color='black', linewidth=1.5),
                      boxprops=dict(facecolor=cmap(2), alpha=0.7))
    ax2.set_xticklabels(labels)
    # Linear regression on per-iteration medians and maxima
    nn_positions = np.arange(1, len(nn_data) + 1, dtype=float)
    x_line = np.array([nn_positions[0], nn_positions[-1]])

    nn_medians = np.array([np.median(d) for d in nn_data])
    slope_med, intercept_med = np.polyfit(nn_positions, nn_medians, 1)
    ax2.plot(x_line, slope_med * x_line + intercept_med, color='red', linewidth=1.5,
             linestyle='--', label=f"Median fit (slope={slope_med:.3g})")

    nn_upper_whiskers = np.array([
        np.min([np.percentile(d, 75) + 1.5 * (np.percentile(d, 75) - np.percentile(d, 25)), np.max(d)])
        for d in nn_data
    ])
    slope_max, intercept_max = np.polyfit(nn_positions, nn_upper_whiskers, 1)
    ax2.plot(x_line, slope_max * x_line + intercept_max, color='orange', linewidth=1.5,
             linestyle='--', label=f"Upper whisker fit (slope={slope_max:.3g})")

    ax2.legend(fontsize='x-small')
ax2.set_xlabel("ASG Iteration")
ax2.set_ylabel("NN Distance")
ax2.set_title("NN Distance Train Set distribution")
nn_path = os.path.join(BASE_DIR, "nndist_comparison.pdf")
fig2.savefig(nn_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"Saved to {nn_path}")

# PCA Projection Per Dimension Entropy line plot
fig3, ax3 = plt.subplots(figsize=(6, 5))
if pca_entropy_data:
    ax3.plot(x, pca_entropy_data, marker='o', linewidth=2, markersize=8,
             color=cmap(4), markerfacecolor='white', markeredgewidth=2)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
ax3.set_xlabel("ASG Iteration")
ax3.set_ylabel("PCA Projection Per Dimension Entropy")
ax3.set_title("PCAProjPerDimEntropy across iterations")
ax3.grid(True, alpha=0.3)
pca_entropy_path = os.path.join(BASE_DIR, "pca_entropy_comparison.pdf")
fig3.savefig(pca_entropy_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print(f"Saved to {pca_entropy_path}")

# Generated vs closest train (most distant across all indices)
DEFAULT_TRAIN_SET_PATH = os.path.expanduser("~/scratch/nanophoto/topoptim/fulloptim/images.npy")
if not os.path.exists(DEFAULT_TRAIN_SET_PATH):
    print(f"WARNING: train set not found at {DEFAULT_TRAIN_SET_PATH}, skipping generated_vs_closest_train_all plot.")
else:
    all_images_plot = np.concatenate([imgs for imgs, _ in loaded_data.values()])
    all_foms_plot = np.concatenate([
        foms if foms is not None else np.zeros(imgs.shape[0])
        for imgs, foms in loaded_data.values()
    ])
    compare_fn = CompareToTrainClosestImage(train_set_path=DEFAULT_TRAIN_SET_PATH)
    compare_fn(images=all_images_plot, savepath=BASE_DIR, model_name="all", fom=all_foms_plot)
    print(f"Saved to {os.path.join(BASE_DIR, 'generated_vs_closest_train.pdf')}")

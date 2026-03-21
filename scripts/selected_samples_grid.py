#!/usr/bin/env python
"""Create a grid image from selected_samples_iter_*.pt files."""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image


def make_grid(images: np.ndarray, ncols: int = 10, padding: int = 2) -> np.ndarray:
    """Arrange images into a grid with padding."""
    n, h, w = images.shape
    nrows = (n + ncols - 1) // ncols

    grid_h = nrows * h + (nrows + 1) * padding
    grid_w = ncols * w + (ncols + 1) * padding
    grid = np.ones((grid_h, grid_w), dtype=np.uint8) * 128

    for idx in range(n):
        row, col = divmod(idx, ncols)
        y = padding + row * (h + padding)
        x = padding + col * (w + padding)
        grid[y : y + h, x : x + w] = images[idx]

    return grid


def main():
    parser = argparse.ArgumentParser(description="Grid of selected samples")
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("active_learning_output"),
        help="Directory containing selected_samples_iter_*.pt files",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output image path")
    parser.add_argument("-c", "--cols", type=int, default=16, help="Number of columns")
    parser.add_argument("-n", "--max-samples", type=int, default=None, help="Max samples per iteration")
    parser.add_argument("--per-iter", action="store_true", help="Save a separate grid per iteration")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("selected_samples_iter_*.pt"))
    if not files:
        # Also check iter_*/ subdirectories
        files = sorted(args.input_dir.glob("iter_*/selected_samples_iter_*.pt"))
    if not files:
        print(f"No selected_samples_iter_*.pt found in {args.input_dir}")
        return

    print(f"Found {len(files)} iteration file(s)")

    all_samples = []
    for f in files:
        samples = torch.load(f, weights_only=False)
        if args.max_samples:
            samples = samples[: args.max_samples]
        print(f"  {f.name}: {samples.shape[0]} samples")

        if args.per_iter:
            imgs = (np.clip(samples.numpy(), 0, 1) * 255).astype(np.uint8)
            grid = make_grid(imgs, ncols=args.cols)
            out = args.input_dir / f"{f.stem}_grid.png"
            Image.fromarray(grid).save(out)
            print(f"  -> {out}")

        all_samples.append(samples)

    # Combined grid
    combined = torch.cat(all_samples, dim=0)
    imgs = (np.clip(combined.numpy(), 0, 1) * 255).astype(np.uint8)
    grid = make_grid(imgs, ncols=args.cols)

    output_path = args.output or args.input_dir / "selected_samples_grid.png"
    Image.fromarray(grid).save(output_path)
    print(f"Saved grid ({combined.shape[0]} samples) -> {output_path}")


if __name__ == "__main__":
    main()

import torch
import numpy as np
from photo_gen.evaluation.eval_single_file import eval_single_file
from pathlib import Path
from icecream import ic
from train import compute_FOM_parallele_safe

def evaluate(images: np.ndarray, savepath: Path = Path("."), fom: np.ndarray | None =
             None) -> dict:
    results = eval_single_file(images=images, savepath=savepath, fom=fom)
    return results

if __name__ == "__main__":
    base = Path(".")

    # Recursively find all selected samples
    sample_files = sorted(base.glob("**/selected_samples_iter_*.pt"))

    if not sample_files:
        print(f"No selected_samples_iter_*.pt found in {base.resolve()}")
        exit(1)

    all_images = []
    for f in sample_files:
        samples = torch.load(f, weights_only=False).cpu().numpy()
        print(f"{f}: {samples.shape[0]} samples")
        all_images.append(samples)

    images = np.concatenate(all_images, axis=0)
    ic(images.shape)
    np.save("new_images.npy", images)

    savepath = Path("eval_results")
    savepath.mkdir(parents=True, exist_ok=True)
    evaluate(images=images, savepath=savepath)



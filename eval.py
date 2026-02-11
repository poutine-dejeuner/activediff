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
    base = Path("active_learning_output/")
    path0 = base / "selected_samples_iter_0.pt"
    images0 = torch.load(path0).cpu().numpy()
    path1 = base / "selected_samples_iter_1.pt"
    images = np.concatenate([images0, torch.load(path1).cpu().numpy()], axis=0)
    ic(images.shape)
    np.save(base / "new_images.npy", images)

    savepath = base / "eval_results"
    savepath.mkdir(parents=True, exist_ok=True)
    evaluate(images=images, savepath=savepath)


    # import argparse
    # from pathlib import Path
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--file", type=str, required=True)
    # parser.add_argument("--savepath", type=str, default=".")
    # args = parser.parse_args()
    # path = Path(args.file)
    # if path.suffix == ".npy":
    #     images = np.load(path)
    # elif path.suffix == ".pt":
    #     images = torch.load(path).cpu().numpy()
    #
    # evaluate(images=images, savepath=args.savepath)



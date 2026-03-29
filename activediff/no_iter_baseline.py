from pathlib import Path
import argparse
import numpy as np
import torch
import activediff
from activediff.utils import compute_distances, dist_select, binarisation, filter_similar_samples
from activediff.models.unet import inference
from activediff.meep_compute_fom import compute_FOM_parallele
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, override

def load_hydra_config(
                    overrides: Optional[List[str]] = None,
                    config_name: str = "config",
                    config_path: str = "configs",
                    version_base: str = "1.3",
                    ) -> DictConfig:
                    with initialize(config_path=config_path,
                    version_base=version_base): 
                            cfg = compose(config_name=config_name,
                                        overrides=overrides or [])
                    return cfg

def main():
        parser = argparse.ArgumentParser(description='No Iteration Baseline')
        parser.add_argument('--checkpoint', type=str, default = 'checkpoints/checkpoint.ckpt',
                            help='Path to the checkpoint file')
        args = parser.parse_args()
        checkpoint_path = args.checkpoint
        savedir = Path('tmp/')
        savedir.mkdir(exist_ok=True)
        cfg = load_hydra_config()
        directory = 'outputs/no_iter_baseline/iter_0'
        Path(directory).mkdir(exist_ok=True)

        images = inference(cfg, checkpoint_path, savepath=str(savedir))
        images = torch.from_numpy(images).float()

        train_set = np.load('data/imagesnorm.npy')
        train_set = torch.from_numpy(train_set).float()
        binar = binarisation(images)
        print(binar.mean(), binar.min(), binar.max())


        # dist select
        distances = compute_distances(images, train_set)
        distance_mask = distances > 10
        images = images[distance_mask]

        # fom select
        fom = compute_FOM_parallele(images)
        if images.shape[0] == 1:
                fom = np.array([fom])
        fom = torch.from_numpy(fom).float()
        fom_idx = fom > 0.48
        fom = fom[fom_idx]
        images = images[fom_idx]

        images, fom = filter_similar_samples(images, fom, distance_threshold=10)
        if images.shape[0] == 0:
                return
        distances = compute_distances(images, train_set)

        torch.save(images, f'{directory}/selected_samples_iter_0.pt')
        torch.save(fom, f'{directory}/selected_fom_scores_iter_0.pt')
        return

if __name__ == "__main__":
        main()

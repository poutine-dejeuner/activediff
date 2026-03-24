from pathlib import Path
import os
import numpy as np
import torch
from activediff.utils import compute_distances, dist_select, fom_select, filter_similar_samples

datapath = Path(os.environ['HOME']) / "drive_scratch/nanophoto/diffusion/train3/7121889"
trainpath = Path(os.environ['HOME']) / "scratch/nanophoto/topoptim/fulloptim/images.npy"
savepath = Path('output/no_iter_dist_select_baseline/iter_0')

images = torch.from_numpy(np.load(datapath / "images.npy")).to(torch.float)
fom = torch.from_numpy(np.load(datapath / "fom.npy")).to(torch.float)
training_data= torch.from_numpy(np.load(trainpath)).to(torch.float)

# Compute distances between all samples
distances = compute_distances(samples=images, training_data=training_data)
dist_mask = distances > 10
images = images[dist_mask]
fom = fom[dist_mask]
fom_mask = fom > 0.48
images = images[fom_mask]
fom = fom[fom_mask]

os.makedirs(savepath, exist_ok=True)
torch.save(images, savepath / "selected_samples_iter_0.pt")
torch.save(fom, savepath / "selected_fom_scores_iter_0.pt")


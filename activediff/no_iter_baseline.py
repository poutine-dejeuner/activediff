from pathlib import Path
import argparse
import numpy as np
import torch
import activediff
from activediff.utils import compute_distances, dist_select, binarisation

parser = argparse.ArgumentParser(description='No Iteration Baseline')
parser.add_argument('directory', type=str, help='Directory containing the data', default='.')
args = parser.parse_args()
directory = args.directory

fom = np.load(f'{directory}/fom.npy')
fom = torch.from_numpy(fom).float()
images = np.load(f'{directory}/images.npy')
images = torch.from_numpy(images).float()
train_set = np.load(Path(activediff.__file__).parent.parent / 'data/imagesnorm.npy')
train_set = torch.from_numpy(train_set).float()
binar = binarisation(images)
print(binar.mean(), binar.min(), binar.max())


# fom select
fom_idx = fom > 0.48
fom = fom[fom_idx]
images = images[fom_idx]

# dist select
distances = compute_distances(images, train_set)
distance_mask = distances > 10
images = images[distance_mask]
distances = distances[distance_mask]
fom = fom[distance_mask]

torch.save(images, f'{directory}/selected_samples_iter_0.pt')
torch.save(fom, f'{directory}/selected_fom_scores_iter_0.pt')


# ActiveDiff

[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-teal.json)](https://github.com/mila-iqia/ResearchTemplate)

Experiment on image generation for nanophotonics by active search with diffusion model. The objective is to generate splitters designs with performance evaluated with EM solver Meep. An initial training set was generated generated with the EM solver and gradient methods. The Active Search generation is then performed in these steps:
For num_epochs:
1. Train a diffusion model on the training data.
2. Generate new samples.
3. Select samples that satisfy sample_distance_to_training_set > distance_threshold.
4. Select samples that satisfy sample_fom > fom_threshold.
5. Add the selected samples to the training ser.

The point of step 3 is to prevent the addition of new samples that are too similar to the initial training set to ensure a growing diversity in the data generation. After only 2 epochs, the process discovers samples that are 

## Installation
Pymeep is needed and is only available by compilation or via conda-forge. The
following steps will install pymeep and the rest of the venv.

```bash
conda create -n mp -c conda-forge pymeep
uv venv --system-site-packages
source .venv/bin/activate
uv sync --extra dev
mkdir data
wget -O data/nanophotodata.zip --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1oRCZmrC0aGuYBmA2VRkR9rROH-XtWnd8'
unzip data/nanophotodata.zip -d data
```

Or just run
```bash
install.sh
```

## Usage

```console
. .venv/bin/activate
python reg_transfo/main.py --help
```

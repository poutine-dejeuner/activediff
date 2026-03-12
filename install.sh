#! /usr/bin/bash
cd ~
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source miniconda3/bin/activate
conda create -n mp python=3.12 conda-forge::pymeep
conda activate mp
wget -qO- https://astral.sh/uv/install.sh | sh
source
cd activediff
uv venv --system-site-packages
source .venv/bin/activate
uv sync --extra dev
mkdir data
wget -O data/nanophotodata.zip --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1oRCZmrC0aGuYBmA2VRkR9rROH-XtWnd8'
unzip data/nanophotodata.zip -d data

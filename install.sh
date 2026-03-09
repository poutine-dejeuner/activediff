#! /usr/bin/bash
conda create -n mp -c conda-forge pymeep
uv venv --system-site-packages
source .venv/bin/activate
uv sync --extra dev
mkdir data
wget -O data/nanophotodata.zip --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1oRCZmrC0aGuYBmA2VRkR9rROH-XtWnd8'
unzip data/nanophotodata.zip -d data

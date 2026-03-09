#! /usr/bin/bash
conda create -n mp -c conda-forge pymeep
# conda install gls
uv venv --system-site-packages
source .venv/bin/activate
uv sync --extra dev


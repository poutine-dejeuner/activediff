# reg_transfo

[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-teal.json)](https://github.com/mila-iqia/ResearchTemplate)

Experiment on image generation for nanophotonics. 

## Installation
Pymeep is needed and is only available by compilation or via conda-forge. The
following steps will install pymeep and the rest of the venv.

```bash
conda create -n mp -c conda-forge pymeep
uv venv --system-site-packages
source .venv/bin/activate
uv sync --extra dev
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

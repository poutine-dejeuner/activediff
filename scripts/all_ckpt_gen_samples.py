import argparse
import glob
from hydra import initialize, compose
import activediff
from activediff.models.unet import inference
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Find .ckpt files and print their epoch")
parser.add_argument('directory', type=str, default='.', help='Directory to search for .ckpt files')
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to Hydra config')
args = parser.parse_args()
directory = args.directory
config_path = args.config

# Charger la config Hydra comme dans main.py
configpath = activediff.__file__.replace('__init__.py', '') + "configs"
print(configpath)
with initialize(version_base="1.3", config_path="../activediff/configs"):
    cfg = compose(config_name="config")
cfg.active_learning.n_to_generate=1

checkpoints = glob.glob(directory + 'checkpoint*.ckpt')
for ckpt in checkpoints:
    sample = inference(cfg=cfg, checkpoint_path=ckpt, meep_eval=False, savepath=directory)
    print(f"Generated sample from {ckpt}")
    plt.imshow(sample.numpy())

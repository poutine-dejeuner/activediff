from pathlib import Path
import glob
import argparse
from hydra import initialize, compose
from activediff.models.unet import inference
import matplotlib.pyplot as plt
import activediff

parsetr = argparse.ArgumentParser()
parsetr.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
args = parsetr.parse_args()
checkpoint = Path(args.checkpoint)
directory = checkpoint.parent

configpath = activediff.__file__.replace('__init__.py', '') + "configs"
print(configpath)
with initialize(version_base="1.3", config_path="../activediff/configs"):
    cfg = compose(config_name="config")
cfg.active_learning.n_to_generate=4

sample = inference(cfg=cfg, checkpoint_path=str(checkpoint), meep_eval=False,
                   savepath=str(directory))
print(f"Generated sample from {checkpoint}")

for i in range(sample.shape[0]):
    plt.imshow(sample[i])
    plt.savefig(directory / f"sample_{i}_from_{checkpoint.stem}.png")
    plt.show()


from pathlib import Path
import glob
import argparse
import os

from pytorch_lightning import LightningModule

def find_ckpt_files(directory):
    files = glob.glob(directory + '*checkpoint*')
    print(files)
    return files

def print_epoch_from_ckpt(ckpt_path):
    model = LightningModule.load_from_checkpoint(ckpt_path)
    epoch = getattr(model, 'current_epoch', None)
    print(f"{ckpt_path}: epoch = {epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find .ckpt files and print their epoch")
    parser.add_argument('-d', '--directory', type=str, default='.',
                        help='Directory to search for .ckpt files')
    args = parser.parse_args()
    directory = args.directory
    for ckpt_file in find_ckpt_files(directory):
        print_epoch_from_ckpt(ckpt_file)

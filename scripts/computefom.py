import numpy as np
from activediff.meep_compute_fom import compute_FOM_parallele

path = "/home/vincent/drive_scratch/nanophoto/diffusion/train3/7121889/images.npy"
images = np.load(path)
fom = compute_FOM_parallele(images)
np.save("/home/vincent/drive_scratch/nanophoto/diffusion/train3/7121889/fom.npy", fom)
print(fom)

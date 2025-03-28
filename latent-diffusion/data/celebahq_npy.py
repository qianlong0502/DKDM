import os
from tqdm import tqdm
from PIL import Image
import numpy as np

folder = "celebahq"
os.makedirs(folder, exist_ok=True)
relpaths = list(range(30000))
relpaths.sort()
relpaths = [f"{folder}/imgHQ{i:05d}.npy" for i in relpaths]

root = "path/to/dataset/CelebA-HQ/data1024x1024"
files = os.listdir(root)
files = [os.path.join(root, i) for i in files]
files = sorted(files)

use_parallel = False
if not use_parallel:
    idx = 0
    for idx in tqdm(range(len(files)), total=len(files)):
        file = files[idx]
        relpath = relpaths[idx]
        img = Image.open(file)
        img = np.array(img)
        # -> 1, 3, 256, 256
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis]
        np.save(relpath, img)
else:
    # accelerate through multiple process
    import concurrent.futures
    def process(idx):
        file = files[idx]
        relpath = relpaths[idx]
        img = Image.open(file)
        img = np.array(img)
        # -> 1, 3, 256, 256
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis]
        np.save(relpath, img)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process, range(len(files)))

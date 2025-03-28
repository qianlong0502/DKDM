import numpy as np
from tqdm import tqdm
import albumentations
from PIL import Image

with open("celebahqtrain.txt", 'r') as f:
    paths = f.readlines()
paths = [f"celebahq/{i.strip()}" for i in paths]
paths.sort()
size = 256
rescaler = albumentations.SmallestMaxSize(max_size = size)
cropper = albumentations.CenterCrop(height=size,width=size)
preprocessor = albumentations.Compose([rescaler, cropper])

use_parallel = False
if not use_parallel:
    npzs = []
    for path in tqdm(paths):
        image = np.load(path).squeeze(0)
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = preprocessor(image=image)["image"]
        npzs.append(image)
else:
    # speed up for loop through multiple process
    import concurrent.futures
    def process(path):
        image = np.load(path).squeeze(0)
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = preprocessor(image=image)["image"]
        return image
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        npzs = list(tqdm(executor.map(process, paths), total=len(paths)))

npzs = np.stack(npzs)
np.savez("celebahq_train.npz", npzs)

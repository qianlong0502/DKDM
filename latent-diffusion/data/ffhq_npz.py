import numpy as np
from tqdm import tqdm
import albumentations
from PIL import Image
size = 256
rescaler = albumentations.SmallestMaxSize(max_size = size)
cropper = albumentations.CenterCrop(height=size,width=size)
preprocessor = albumentations.Compose([rescaler, cropper])

with open("ffhqtrain.txt", 'r') as f:
    paths = f.readlines()
paths = [f"ffhq/{i.strip()}" for i in paths]
paths.sort()

use_parallel = False
if not use_parallel:
    npzs = []
    for path in tqdm(paths):
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = preprocessor(image=image)["image"]
        npzs.append(image)
else:
    # speed up for loop through multiple process
    import concurrent.futures
    def process(path):
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = preprocessor(image=image)["image"]
        return image
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        npzs = list(tqdm(executor.map(process, paths), total=len(paths)))

npzs = np.stack(npzs)

np.savez("ffhq_train.npz", npzs)


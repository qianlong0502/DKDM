import numpy as np
import os
from tqdm import tqdm

npz1 = "../log/celeba256/samples/00490000/2024-10-03-15-25-42/numpy/10240x256x256x3-samples.npz"
npz2 = "../log/celeba256/samples/00490000/2024-10-03-15-26-49/numpy/10240x256x256x3-samples.npz"
npz3 = "../log/celeba256/samples/00490000/2024-10-03-15-27-55/numpy/10240x256x256x3-samples.npz"
npz1 = np.load(npz1)['arr_0']
npz2 = np.load(npz2)['arr_0']
npz3 = np.load(npz3)['arr_0']

npzs = np.concatenate([npz1, npz2, npz3], axis=0)[:25000]
npzs = [npzs[i] for i in range(0, 25000)]

os.makedirs("celebahq_baseline", exist_ok=True)

for idx, npy in tqdm(enumerate(npzs), total=len(npzs)):
    filename = os.path.join("celebahq_baseline", f"{idx:05d}.npy")
    _npy = npy.transpose(2, 0, 1)
    _npy = np.expand_dims(_npy, 0)
    np.save(filename, _npy)

# with open("celebahq_train_baseline.txt", "w") as f:
#     for idx in range(25000):
#         f.write(f"{idx:05d}.npy\n")

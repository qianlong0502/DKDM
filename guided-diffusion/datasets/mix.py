import os
from PIL import Image
from tqdm import tqdm

import numpy as np
np.random.seed(0)

output_dir = 'cifar10_baseline_0.05'
ratio_dataset = 0.05

dataset_file = "path/to/cifar10_train.npz"
# dataset_file = "path/to/celeba64_train.npz"
# dataset_file = "path/to/ImageNet32_train_all.npz"

sync_file = "path/to/samples_1000t_50000x32x32x3.npz"
# sync_file = "path/to/samples_100t_202599x64x64x3.npz"
# sync_file = "path/to/samples_100t_1281167x32x32x3.npz"

dataset = np.load(dataset_file)['arr_0']

sync = np.load(sync_file)['arr_0']

len_dataset = len(dataset)
len_sync = len(sync)

assert len_dataset == len_sync

dataset_candidate = list(range(len_dataset))
sync_candidate = list(range(len_sync))

ratio_sync = 1 - ratio_dataset

chosen_len_dataset = round(len_dataset * ratio_dataset)
chosen_len_sync = len_dataset - chosen_len_dataset

chosen_dataset = np.random.choice(dataset_candidate, chosen_len_dataset, replace=False)
chosen_sync = np.random.choice(sync_candidate, chosen_len_sync, replace=False)

assert len(set(chosen_dataset)) + len(set(chosen_sync)) == len_dataset

a = dataset[chosen_dataset]
b = sync[chosen_sync]

result = np.concatenate([a, b], axis=0)

os.makedirs(output_dir, exist_ok=True)

k = 0
for i in tqdm(result):
    img = Image.fromarray(i)
    filename = os.path.join(output_dir, str(k).zfill(6)+".png")
    img.save(filename)
    k = k+1

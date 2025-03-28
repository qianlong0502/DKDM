import os
import random
random.seed(0)

txt = 'ffhqtrain_baseline'
ratio_dataset = 1
txt = f"{txt}_{ratio_dataset}.txt"

dataset_root = "ffhq"
sync_file = "ffhq_baseline"

with open("ffhqtrain.txt", 'r') as f:
    dataset_files = f.readlines()
    dataset_files = sorted([os.path.join(dataset_root, i.strip()) for i in dataset_files])
for path in dataset_files:
    assert os.path.exists(path) and os.path.isfile(path)

dataset_files = sorted(dataset_files)

with open("ffhqtrain_baseline.txt", 'r') as f:
    sync_files = f.readlines()
    sync_files = sorted([os.path.join(sync_file, i.strip()) for i in sync_files])
for path in sync_files:
    assert os.path.exists(path) and os.path.isfile(path)
sync_file = sorted(sync_files)

len_dataset = len(dataset_files)
chosen_len_dataset = round(len_dataset * ratio_dataset)
chosen_len_sync = len_dataset - chosen_len_dataset
chosen_dataset = random.sample(dataset_files, chosen_len_dataset)
chosen_sync = random.sample(sync_files, chosen_len_sync)

assert len(set(chosen_dataset)) + len(set(chosen_sync)) == len_dataset

mix_dataset_files = chosen_dataset + chosen_sync

with open(txt, 'w') as f:
    for path in mix_dataset_files:
        f.write(path + '\n')

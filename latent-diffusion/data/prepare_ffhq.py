import os
from tqdm import tqdm

folders = [
    "../log/ffhq256/samples/00634478/2024-10-10-12-28-42/img",
    "../log/ffhq256/samples/00634478/2024-10-10-15-42-36/img",
    "../log/ffhq256/samples/00634478/2024-10-16-15-06-24/img",
]
files = []
for path in folders:
    files += [os.path.join(path, f) for f in os.listdir(path)]

files.sort()
for path in files: assert os.path.exists(path)

files = files[:60000]

cnt = 0
for path in tqdm(files):
    os.rename(path, f"ffhq_baseline/{cnt:05d}.png")
    cnt += 1

# with open("ffhqtrain_baseline.txt", 'w') as f:
#     for idx in range(60000):
        # f.write(f"{idx:05d}.png\n")

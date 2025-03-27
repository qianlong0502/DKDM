import os
import cv2
import numpy as np

def imgs_to_npz():
    npz = []

    for img in os.listdir("./cifar"):
        img_arr = cv2.imread("./cifar/" + img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        npz.append(img_arr)

    output_npz = np.array(npz)
    np.savez('cifar10_train.npz', output_npz)
    print(f"{output_npz.shape} size array saved into cifar10_train.npz")

if __name__ == '__main__':
    imgs_to_npz()
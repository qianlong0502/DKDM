import os
import torch
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

class FaceEmptyTest(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()

        self.data = [{"image":np.ones((256,256,3))}] * size
        self.keys = keys

class CelebAHQTrain(FacesBase):
    def __init__(self, size, root="data/celebahq", txt_file="data/celebahqtrain.txt", keys=None):
        super().__init__()
        root = root
        with open(txt_file, "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths] 
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys

class CelebAHQTrainRepeat(FacesBase):
    def __init__(self, size, keys=None, repeat=10):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        self.repeat = repeat
        paths = [os.path.join(root, relpath) for relpath in relpaths] * self.repeat
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CelebAHQValidationRepeat(FacesBase):
    def __init__(self, size, keys=None, repeat=50):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        self.repeat = repeat
        paths = [os.path.join(root, relpath) for relpath in relpaths] * repeat
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQTrain(FacesBase):
    def __init__(self, size, root="data/images1024x1024", txt="data/ffhqtrain.txt", keys=None):
        super().__init__()
        with open(txt, "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys

class FFHQTrainRepeat(FacesBase):
    def __init__(self, size, keys=None, repeat=3):
        super().__init__()
        root = "data/images1024x1024"
        with open("data/ffhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        self.repeat = repeat
        paths = [os.path.join(root, relpath) for relpath in relpaths] * self.repeat
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys

class FFHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/images1024x1024"
        with open("data/ffhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FacesHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQTrain(size=size, keys=keys)
        d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQValidation(size=size, keys=keys)
        d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex
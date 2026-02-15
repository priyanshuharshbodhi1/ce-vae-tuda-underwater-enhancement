"""
Dataset for loading unpaired real underwater images (no ground truth needed).
Used during TUDA feature-level alignment training to bring real-world
underwater distributions into the CE-VAE encoder's feature space.
"""
import os
import glob
from typing import List, Union

import cv2
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_img(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    return image


class UnpairedRealUnderwaterDataset(Dataset):
    """
    Loads unpaired real underwater images for domain adaptation.
    
    Supports two modes:
    1. List file mode: reads image paths from a text file (one path per line)
    2. Directory mode: recursively finds all images in a directory
    
    Images are resized, randomly cropped/flipped, and normalized to [-1, 1].
    """

    def __init__(self, 
                 images_list_file: Union[str, List[str]] = None,
                 images_dir: str = None,
                 size: int = 256,
                 random_crop: bool = True,
                 random_flip: bool = True,
                 max_size: int = None):
        super().__init__()
        
        self.size = size
        self.paths = []
        
        # Mode 1: Load from list file(s)
        if images_list_file is not None:
            if isinstance(images_list_file, str):
                images_list_file = [images_list_file]
            for list_file in images_list_file:
                with open(list_file, "r") as f:
                    self.paths += [line.strip() for line in f.readlines() if line.strip()]
        
        # Mode 2: Load from directory
        if images_dir is not None:
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
            for ext in extensions:
                self.paths += glob.glob(os.path.join(images_dir, '**', ext), recursive=True)
        
        if len(self.paths) == 0:
            raise ValueError(
                "No images found. Provide either images_list_file or images_dir."
            )
        
        # Build augmentation pipeline
        if max_size is None:
            max_size = size
            
        augmentations = [
            albumentations.SmallestMaxSize(max_size=max_size),
        ]
        
        if random_crop:
            augmentations.append(albumentations.RandomCrop(height=size, width=size))
        else:
            augmentations.append(albumentations.CenterCrop(height=size, width=size))
        
        if random_flip:
            augmentations.append(albumentations.HorizontalFlip(p=0.5))
        
        self.preprocessor = albumentations.Compose(augmentations)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = _load_img(self.paths[i])
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return {"image": image, "file_path_": self.paths[i]}

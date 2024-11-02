import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .preprocess import preprocess  # импорт функции предобработки

class CTScanDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, sigma=1):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_filenames = os.listdir(images_dir)
        self.transform = transform
        self.sigma = sigma

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert('L')
        image = np.array(image)
        processed_image = preprocess(image, sigma=self.sigma)

        mask = np.zeros(processed_image.shape, dtype=np.uint8)
        for label, suffix in zip([1, 2, 3], ['-AO', '-COR', '-PA']):
            mask_path = os.path.join(self.masks_dir, image_name.replace('.jpg', f"{suffix}.png"))
            if os.path.exists(mask_path):
                mask_layer = Image.open(mask_path)
                mask_layer = np.array(mask_layer)
                mask[mask_layer > 0] = label

        if self.transform:
            processed_image = self.transform(Image.fromarray(processed_image))
            mask = self.transform(Image.fromarray(mask))

        return processed_image, mask

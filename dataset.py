import os
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class PdccaDataset(Dataset):
	def __init__(self, image_dir, mask_dir, transform=None):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.transform = transform
		self.images = sorted(os.listdir(image_dir))
		self.masks = sorted(os.listdir(mask_dir))

	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, index):
		img_path = os.path.join(self.image_dir, self.images[index])
		mask_path = os.path.join(self.mask_dir, self.masks[index])
		image = np.load(img_path)
		mask = np.load(mask_path)
		mask[mask > 0] = 1

		if self.transform is not None:
			augmentations = self.transform(image=image, mask=mask)
			image = augmentations["image"]
			mask = augmentations["mask"]

		#Input channel for unet, unet++, and transunet is kept as 1. 
		image = np.resize(image, (1, image.shape[0], image.shape[1]))
		mask = np.resize(mask, (1, mask.shape[0], mask.shape[1]))

		return image, mask

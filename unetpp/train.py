import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from unetpp import NestedUNet
from utils import(
	load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs,
	)

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 300 #Change to 1000 if training the optic chiasm
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False #Change to true if saved model is available

#Change the directory names to train on a different strucutre
TRAIN_IMG_DIR = "../data_brainstem/train_images"
TRAIN_MASK_DIR = "../data_brainstem/train_masks"
VAL_IMG_DIR = "../data_brainstem/val_images"
VAL_MASK_DIR = "../data_brainstem/val_masks"

#Set names according to the strcuture being trained
CHECKPOINT = "my_checkpoint_brainstem.pth.tar"
IMAGE_FOLDER = "saved_images_brainstem"

def train_fn(loader, model, optimizer, loss_fn, loss_values, scaler):
	loop = tqdm(loader)

	for batch_idx, (data, targets) in enumerate(loop):
		data = data.to(device=DEVICE)
		targets = targets.to(device=DEVICE)	#Might need to change

		#forward
		with torch.cuda.amp.autocast():
			predictions = model(data)
			loss = loss_fn(predictions, targets)

		#backward
		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		loop.set_postfix(loss=loss.item())
	loss_values.append(loss.item())

def main():
	train_transform = A.Compose(
		[
			A.Rotate(limit=35, p=1.0),
			A.HorizontalFlip(p=0.5),
			A.VerticalFlip(p=0.1),
		],
	)

	val_transform = A.Compose(
		[
		]
	)

	model = NestedUNet(in_channels=1, out_channels=1).to(DEVICE)

	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	train_loader, val_loader = get_loaders(
		TRAIN_IMG_DIR,
		TRAIN_MASK_DIR,
		VAL_IMG_DIR,
		VAL_MASK_DIR,
		BATCH_SIZE,
		train_transform,
		NUM_WORKERS,
		PIN_MEMORY,
	)

	if LOAD_MODEL:
		load_checkpoint(torch.load(CHECKPOINT), model)

	check_accuracy(val_loader, model, device=DEVICE)
	scaler = torch.cuda.amp.GradScaler()

	loss_values = []
	for epoch in range(NUM_EPOCHS):
		train_fn(train_loader, model, optimizer, loss_fn, loss_values, scaler)

		checkpoint = {
			"state_dict": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}
		save_checkpoint(checkpoint, CHECKPOINT)

		check_accuracy(val_loader, model, device=DEVICE)

		save_predictions_as_imgs(
			val_loader, model, folder=IMAGE_FOLDER, device=DEVICE
		)

	plt.plot(np.array(loss_values), 'r')
	plt.savefig('unetpp_loss_plot.png')

if __name__ == "__main__":
	main()

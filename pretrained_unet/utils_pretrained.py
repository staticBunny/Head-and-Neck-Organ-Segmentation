import torch
import torchvision
import matplotlib.pyplot as plt
from dataset_pretrained import PdccaDataset
from torch.utils.data import DataLoader
from skimage import segmentation, color
from scipy.spatial.distance import directed_hausdorff

def save_checkpoint(state, filename):
	print("=> Saving chekpoint")
	torch.save(state, filename)

def load_checkpoint(checkpoint, model):
	print("=> Loading checkpoint")
	model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
	train_dir,
	train_maskdir,
	val_dir,
	val_maskdir,
	batch_size,
	train_transform,
	num_workers=4,
	pin_memory=True,
):
	'''
	returns training and validatation images (as stored in PdccaDataset) in batches
	'''

	train_ds = PdccaDataset(
		image_dir=train_dir,
		mask_dir=train_maskdir,
		transform = train_transform,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=False,
	)

	val_ds = PdccaDataset(
		image_dir = val_dir,
		mask_dir = val_maskdir,
		transform = None,
	)

	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=False,
	)

	return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
	'''
	prints the dice score coefficient (DSC) and Hausdorff distance (HD) of the model prediction
	'''

	num_correct = 0
	num_pixels = 0
	dice_score = 0
	hd = 0
	model.eval()

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			preds = model(x)
			preds = (preds > 0.5).float()
			num_correct += (preds == y).sum()
			num_pixels += torch.numel(preds)
			dice_score += (2 * (preds * y).sum()) / ((preds +y).sum() + 1e-8)
			for i in range(y.shape[0]):
				hd += directed_hausdorff(y[i][0].cpu(), preds[i][0].cpu())[0]
			hd = hd/y.shape[0]
	
	print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
	print(f"Dice score: {dice_score/len(loader)}")
	print(f"Hausdorff distance: {hd/len(loader)}")
	model.train()

def save_predictions_as_imgs(
	loader, model, folder, device="cuda"
):
	model.eval()
	for idx, (x, y) in enumerate(loader):
		x = x.to(device=device)
		with torch.no_grad():
			preds = model(x)
			preds = (preds > 0.5).float()

		torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
		torchvision.utils.save_image(y, f"{folder}/main_{idx}.png")
		torchvision.utils.save_image(x, f"{folder}/image_{idx}.png")
	
	model.train()

def dice_loss(pred, target):
	'''
	custom function to get dice score coefficient
	'''

	pred = torch.sigmoid(pred)

	pred = pred.contiguous().view(-1)
	target = target.contiguous().view(-1)

	intersection = torch.sum(pred * target)
	pred_sum = torch.sum(pred * pred)
	target_sum = torch.sum(target * target)

	return 1 - ((2. * intersection + 1e-5) / (pred_sum + target_sum + 1e-5))

def loss_plot(loss, epochs):
	'''
	plots the model loss over number of epochs
	'''

	plt.plot(epoch, loss, 'g') 
	plt.title('Training Loss')
	plt.xlabel('Epochs')
	plt.ylable('Loss')
	plt.legend()
	plt.show()

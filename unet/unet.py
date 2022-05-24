import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
	# in_channels = number of channels in the original image
	# out_channels = number of filters to use in 1st conv
	# out_channels = number of filters to use in 2st conv
	def __init__(self, in_channels, out_channels, stride = 1):
		super(DoubleConv, self).__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.double_conv(x)

class UNet(nn.Module):
	def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512, 1028]):
		super(UNet, self).__init__()

		self.pool = nn.MaxPool2d(2, 2)
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

		self.conv0_0 = DoubleConv(in_channels, features[0])
		self.conv1_0 = DoubleConv(features[0], features[1])
		self.conv2_0 = DoubleConv(features[1], features[2])
		self.conv3_0 = DoubleConv(features[2], features[3])
		self.conv4_0 = DoubleConv(features[3], features[4])

		self.conv3_1 = DoubleConv(features[3]+features[4], features[3])
		self.conv2_2 = DoubleConv(features[2]+features[3], features[2])
		self.conv1_3 = DoubleConv(features[1]+features[2], features[1])
		self.conv0_4 = DoubleConv(features[0]+features[1], features[0])

		self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)


	def forward(self, input):
		x0_0 = self.conv0_0(input)
		x1_0 = self.conv1_0(self.pool(x0_0))
		x2_0 = self.conv2_0(self.pool(x1_0))
		x3_0 = self.conv3_0(self.pool(x2_0))
		x4_0 = self.conv4_0(self.pool(x3_0))

		x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
		x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
		x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
		x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

		output = self.final(x0_4)
		return output


def test():
	x = torch.randn((3, 1, 160, 160))
	model = UNet(in_channels=1, out_channels=1)
	preds = model(x)
	print(preds.shape)
	print(x.shape)
	assert preds.shape == x.shape

if __name__ == "__main__":
	test()

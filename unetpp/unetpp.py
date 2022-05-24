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

class NestedUNet(nn.Module):
	def __init__(self,
		in_channels,
		out_channels, 
		features = [64, 128, 256, 512, 1024],
		deep_supervision=False
	):
		super(NestedUNet, self).__init__()

		self.deep_supervision = deep_supervision

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

		self.conv0_0 = DoubleConv(in_channels, features[0])
		self.conv1_0 = DoubleConv(features[0], features[1])
		self.conv2_0 = DoubleConv(features[1], features[2])
		self.conv3_0 = DoubleConv(features[2], features[3])
		self.conv4_0 = DoubleConv(features[3], features[4])

		self.conv0_1 = DoubleConv(features[0] + features[1], features[0])
		self.conv1_1 = DoubleConv(features[1] + features[2], features[1])
		self.conv2_1 = DoubleConv(features[2] + features[3], features[2])
		self.conv3_1 = DoubleConv(features[3] + features[4], features[3])

		self.conv0_2 = DoubleConv(features[0]*2 + features[1], features[0])
		self.conv1_2 = DoubleConv(features[1]*2 + features[2], features[1])
		self.conv2_2 = DoubleConv(features[2]*2 + features[3], features[2])

		self.conv0_3 = DoubleConv(features[0]*3 + features[1], features[0])
		self.conv1_3 = DoubleConv(features[1]*3 + features[2], features[1])

		self.conv0_4 = DoubleConv(features[0]*4 + features[1], features[0])

		self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

	
	def forward(self, x):
		x0_0 = self.conv0_0(x)
		x1_0 = self.conv1_0(self.pool(x0_0))
		x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

		x2_0 = self.conv2_0(self.pool(x1_0))
		x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
		x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

		x3_0 = self.conv3_0(self.pool(x2_0))
		x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
		x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
		x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

		x4_0 = self.conv4_0(self.pool(x3_0))
		x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
		x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
		x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
		x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

		if self.deep_supervision:
			out1 = self.final(x0_1)
			out2 = self.final(x0_2)
			out3 = self.final(x0_3)
			out4 = self.final(x0_4)
			return [out1, out2, out3, out4]
		
		out = self.final(x0_4)
		return out


def test():
	x = torch.randn((3, 1, 160, 160))
	model = NestedUNet(in_channels=1, out_channels=1)
	preds = model(x)
	print(preds.shape)
	print(x.shape)
	assert preds.shape == x.shape

if __name__ == "__main__":
	test()

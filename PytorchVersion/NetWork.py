import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable


def same_padding(tensor, ksizes=[3, 3], strides=[1, 1], rates=[1, 1]):
	"""
	 @param tensor: tensor that needs to be added to SP
	 @param ksizes: Convolution kernel size list
	 @param strides: list of convolution steps
	 @param rates: Hollow convolution feeling size list
	"""
	assert len(tensor.size()) == 4
	batch_size, channel, rows, cols = tensor.size()
	out_rows = (rows + strides[0] - 1) // strides[0]
	out_cols = (cols + strides[1] - 1) // strides[1]
	effective_k_row = (ksizes[0] - 1) * rates[0] + 1
	effective_k_col = (ksizes[1] - 1) * rates[1] + 1
	padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
	padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
	padding_top = int(padding_rows / 2.)
	padding_left = int(padding_cols / 2.)
	padding_bottom = padding_rows - padding_top
	padding_right = padding_cols - padding_left
	paddings = (padding_left, padding_right, padding_top, padding_bottom)
	tensor = nn.ZeroPad2d(paddings)(tensor)
	return tensor


class InBlock(nn.Module):
	def __init__(self, input_channels, output_channels, kernel_size):
		"""
		The first convolution block
		 @param input_channels: Number of input channels
		 @param output_channels: Number of output channels
		 @param kernel_size: Convolution kernel size
		"""
		super(InBlock, self).__init__()
		self.op = nn.Sequential(
			nn.Conv2d(in_channels=input_channels,
			          out_channels=output_channels,
			          kernel_size=kernel_size,
			          stride=1,
			          padding=0,
			          ),
			nn.BatchNorm2d(output_channels,
			               eps=1e-5,
			               momentum=0.01,
			               affine=False,
			               ),
			nn.ReLU(inplace=False)
		)

	def forward(self, tensor, sp_ksize=[3, 3]):
		tensor = same_padding(tensor, sp_ksize)
		return self.op(tensor)


class BaseBlock(nn.Module):
	def __init__(self, channel_num, kernel_size):
		"""
		Residual block
		 @param channel_num: number of channels
		 @param kernel_size: Convolution kernel size
		"""
		super(BaseBlock, self).__init__()
		self.op1 = nn.Sequential(
			nn.Conv2d(in_channels=channel_num,
			          out_channels=channel_num,
			          kernel_size=kernel_size,
			          stride=1,
			          padding=0, ),
			nn.BatchNorm2d(channel_num,
			               eps=1e-5,
			               affine=False,
			               momentum=0.01,
			               ),
			nn.ReLU(inplace=False)
		)
		self.op2 = nn.Sequential(
			nn.Conv2d(in_channels=channel_num,
			          out_channels=channel_num,
			          kernel_size=kernel_size,
			          stride=1,
			          padding=0, ),
			nn.BatchNorm2d(channel_num,
			               eps=1e-5,
			               affine=False,
			               momentum=0.01,
			               ),
		)
		self.acf = nn.ReLU(inplace=False)

	def forward(self, tensor):
		tensor_to_conv = same_padding(tensor)
		part_out = same_padding(self.op1(tensor_to_conv))
		return self.acf(torch.add(self.op2(part_out), tensor))


class ResBlock(nn.Module):
	def __init__(self, block_num, channel_num, res_block_channel_size):
		"""
		Residual network
		 @param block_num: the number of residual blocks
		 @param channel_num: number of channels
		 @param res_block_channel_size: the size of the convolution kernel of the residual block
		"""
		super(ResBlock, self).__init__()
		self.blocks = nn.ModuleList()
		for _ in range(block_num):
			block = BaseBlock(channel_num, res_block_channel_size)
			self.blocks += [block]

	def forward(self, tensor):
		for each in self.blocks:
			tensor = each(tensor)

		return tensor


class FullConnect(nn.Module):
	def __init__(self, channel_num, policy_num):
		"""
		Fully connected layer
		 @param channel_num: number of channels
		 @param policy_num: the number of output policies
		"""
		super(FullConnect, self).__init__()
		self.op = nn.Linear(channel_num, policy_num)

	def forward(self, tensor):
		return self.op(tensor)


class Policy(nn.Module):
	def __init__(self, arg, is_exp=False):
		"""
		Policy network outputs 361 placement positions and the probability of 1 pass
		 @param arg: parameter list
		"""
		super(Policy, self).__init__()
		self.bath_size = arg.MINI_BATCH_SIZE
		self.ways = arg.ways
		if not is_exp:
			arg.channels_num = arg.channels_num // 2
		self.input_block = InBlock(arg.input_block_channels,
		                           arg.channels_num,
		                           arg.input_block_channel_size
		                           )
		self.res_block = ResBlock(arg.block_num,
		                          arg.channels_num,
		                          arg.res_block_channel_size,
		                          )
		self.p_conv = InBlock(arg.channels_num,
		                      arg.p_fc_channels,
		                      arg.fc_channel_size,
		                      )
		self.v_conv = InBlock(arg.channels_num,
		                      arg.v_fc_channels,
		                      arg.fc_channel_size,
		                      )
		self.fc = FullConnect(arg.ways ** 2 * 2, arg.policy_num)
		self.v_fc_1 = FullConnect(arg.ways ** 2, arg.v_fc_nums)
		self.v_fc_2 = FullConnect(arg.v_fc_nums, 1)
		self.ac1 = nn.ReLU()
		self.ac2 = nn.Tanh()
		if is_exp:
			print("Load weight:.")
			weights = scio.loadmat('weight.mat')
			self.input_block.op[0].weight = torch.nn.Parameter(
				torch.Tensor(np.transpose(weights['Variable'], (3, 2, 0, 1))))
			self.input_block.op[0].bias = torch.nn.Parameter(torch.Tensor(weights['Variable_1'].reshape((-1,))))
			self.input_block.op[1].running_mean = Variable(
				torch.tensor(weights['bn0/batch_normalization/moving_mean'].reshape((-1,)), requires_grad=True))
			self.input_block.op[1].running_var = Variable(
				torch.tensor(weights['bn0/batch_normalization/moving_variance'].reshape((-1,)), requires_grad=True))

			vnum = 2
			bnum = 1
			for i in range(arg.block_num):
				self.res_block.blocks[i].op1[0].weight = torch.nn.Parameter(
					torch.Tensor(np.transpose(weights['Variable_' + str(vnum)], (3, 2, 0, 1))))
				vnum += 1
				self.res_block.blocks[i].op1[0].bias = torch.nn.Parameter(
					torch.Tensor(weights['Variable_' + str(vnum)].reshape((-1,))))
				vnum += 1
				self.res_block.blocks[i].op1[1].running_mean = Variable(
					torch.tensor(weights['bn' + str(bnum) + '/batch_normalization/moving_mean'].reshape((-1,)),
					             requires_grad=True))
				self.res_block.blocks[i].op1[1].running_var = Variable(
					torch.tensor(weights['bn' + str(bnum) + '/batch_normalization/moving_variance'].reshape((-1,)),
					             requires_grad=True))
				bnum += 1
				self.res_block.blocks[i].op2[0].weight = torch.nn.Parameter(
					torch.Tensor(np.transpose(weights['Variable_' + str(vnum)], (3, 2, 0, 1))))
				vnum += 1
				self.res_block.blocks[i].op2[0].bias = torch.nn.Parameter(
					torch.Tensor(weights['Variable_' + str(vnum)].reshape((-1,))))
				vnum += 1
				self.res_block.blocks[i].op2[1].running_mean = Variable(
					torch.tensor(weights['bn' + str(bnum) + '/batch_normalization/moving_mean'].reshape((-1,)),
					             requires_grad=True))
				self.res_block.blocks[i].op2[1].running_var = Variable(
					torch.tensor(weights['bn' + str(bnum) + '/batch_normalization/moving_variance'].reshape((-1,)),
					             requires_grad=True))
				bnum += 1

			self.p_conv.op[0].weight = torch.nn.Parameter(
				torch.Tensor(np.transpose(weights['Variable_26'], (3, 2, 0, 1))))
			self.p_conv.op[0].bias = torch.nn.Parameter(torch.Tensor(weights['Variable_27'].reshape((-1,))))
			self.p_conv.op[1].running_mean = Variable(
				torch.tensor(weights['bn13/batch_normalization/moving_mean'].reshape((-1,)), requires_grad=True))
			self.p_conv.op[1].running_var = Variable(
				torch.tensor(weights['bn13/batch_normalization/moving_variance'].reshape((-1,)), requires_grad=True))

			self.fc.op.weight = torch.nn.Parameter(torch.Tensor(np.transpose(weights['Variable_28'], (1, 0))))
			self.fc.op.bias = torch.nn.Parameter(torch.Tensor(weights['Variable_29'].reshape((-1,))))

			self.v_conv.op[0].weight = torch.nn.Parameter(
				torch.Tensor(np.transpose(weights['Variable_30'], (3, 2, 0, 1))))
			self.v_conv.op[0].bias = torch.nn.Parameter(torch.Tensor(weights['Variable_31'].reshape((-1,))))
			self.v_conv.op[1].running_mean = Variable(
				torch.tensor(weights['bn14/batch_normalization/moving_mean'].reshape((-1,)), requires_grad=True))
			self.v_conv.op[1].running_var = Variable(
				torch.tensor(weights['bn14/batch_normalization/moving_variance'].reshape((-1,)), requires_grad=True))

			self.v_fc_1.op.weight = torch.nn.Parameter(torch.Tensor(np.transpose(weights['Variable_32'], (1, 0))))
			self.v_fc_1.op.bias = torch.nn.Parameter(torch.Tensor(weights['Variable_33'].reshape((-1,))))

			self.v_fc_2.op.weight = torch.nn.Parameter(torch.Tensor(np.transpose(weights['Variable_34'], (1, 0))))
			self.v_fc_2.op.bias = torch.nn.Parameter(torch.Tensor(weights['Variable_35'].reshape((-1,))))

			print("The weights are loaded")

	def forward(self, input_tensor):
		# pytorch has no SP
		# Pass the SP before convolution and then set the padding in pytorch to 0
		output = self.input_block(input_tensor)
		output = self.res_block(output)

		p_output = self.p_conv(output, sp_ksize=[1, 1])
		p_output = torch.reshape(p_output, (-1, self.ways ** 2 * 2))
		p_output = self.fc(p_output)

		v_output = self.v_conv(output, sp_ksize=[1, 1])
		v_output = torch.reshape(v_output, (-1, self.ways ** 2))
		v_output = self.v_fc_1(v_output)
		v_output = self.ac1(v_output)
		v_output = self.v_fc_2(v_output)
		v_output = self.ac2(v_output)

		return p_output, v_output


class Critic(Policy):
	def __init__(self, arg):
		super().__init__(arg=arg)
		# Enter two more dimensions
		"""
			One more dimension is the reshape of the placement position of the p
			The second is all 0 if p[362]==0 else all 1
		"""
		self.input_block = InBlock(arg.input_block_channels + 2,
		                           arg.channels_num,
		                           arg.input_block_channel_size)
		self.res_block = ResBlock(arg.block_num + 2,
		                          arg.channels_num,
		                          arg.res_block_channel_size,
		                          )
		self.fc = FullConnect(arg.ways ** 2 * 2, 1)

	def forward(self, input_tensor):
		output = self.input_block(input_tensor)
		output = self.res_block(output)
		p_output = self.p_conv(output, sp_ksize=[1, 1])
		p_output = torch.reshape(p_output, (-1, self.ways ** 2 * 2))
		p_output = self.fc(p_output)
		return p_output

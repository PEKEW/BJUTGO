# That's not a good idea
class Args:
	def __init__(self):
		self.seed = 1
		self.gpu_id = 0
		self.ways = 19
		self.input_block_channels = 18
		self.channels_num = 128
		self.input_block_channel_size = 3
		self.block_num = 6
		self.res_block_channel_size = 3
		self.p_fc_channels = 2
		self.v_fc_channels = 1
		self.v_fc_nums = 256
		self.fc_channel_size = 1
		self.policy_num = self.ways ** 2 + 1
		self.lr = 1e-5
		self.weight_decay = 1.0e-4
		self.epochs = 100
		self.DATA_ITEM_LINES = 16 + 1 + 1 + 1
		self.MINI_BATCH_SIZE = 512
		self.MAX_STEP_NUM = 200000
		self.save_path = 'model'
		self.momentum = 0.9
		self.max_running_time = 10
		self.cuda_str = 'cuda:' + str(self.gpu_id)

	"""
		print() is the same as info()
		 The difference is that print directly prints without returning
		 info directly returns the parameter dictionary
	"""

	def print(self):
		print(self.__dict__)

	def info(self):
		return self.__dict__

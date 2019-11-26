import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Sampler
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable
import numpy as np


class DataSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class MyAdaptiveLR(object):
	# --- Reduce LR when cost curve flattens --- 

	def __init__(self, optimizer, factor=0.1, patience=50, stag_range=1e-4, cooldown=0, min_lr=0):
		
		super(MyAdaptiveLR, self).__init__()

		self.optimizer = optimizer  # Imported

		self.factor = factor  # Factor should be < 1.0
		self.min_lr = min_lr
		self.stag_range = stag_range  # Stagnation Range
		self.stag_count = 0  # Number of epochs that have stagnated
		self.patience = patience # Number of epochs allowed with no improvement
		self.cooldown = cooldown  # Number of epochs to wait before resuming
		self.cooldown_counter = 0
		self.monitor_op = None
		self.lowest_val = np.Inf  

		self.optimizer = optimizer

	def _compare_new_loss(self,new_loss):
		change_minimum = self.lowest_val - self.stag_range # Tests whether it has decreased enough
		return new_loss < change_minimum  # If new loss if below the stangation bound


	def step(self, loss, epoch):
		new_loss = loss

		if self.in_cooldown():
			self.cooldown_counter -= 1
			self.stag_count = 0

		if self._compare_new_loss(new_loss):
			self.lowest_val = new_loss
			self.stag_count = 0
		
		elif not self.in_cooldown():
			if self.stag_count >= self.patience:  # Iff
				for param_group in self.optimizer.param_groups:
					old_lr = float(param_group['lr'])

					if old_lr > self.min_lr:
						new_lr = old_lr * self.factor
						new_lr = max(new_lr, self.min_lr)
						param_group['lr'] = new_lr

						print('\nEpoch %d: reducing learning rate to %s.' % (epoch, new_lr))
						self.cooldown_counter = self.cooldown
						self.stag_count = 0
			
			self.stag_count += 1
	

	def in_cooldown(self):
		return self.cooldown_counter > 0


class PruneLinear(nn.Linear):
	def __init__(self, in_features, out_features, bias=True):
		super(PruneLinear, self).__init__(in_features, out_features, bias)
		self.mask_flag = False

	def set_mask(self, mask):
		self.register_buffer('mask', mask)
		mask_var = self.get_mask()
		self.weight.data = self.weight.data*mask_var.data
		self.mask_flag = True

	def get_mask(self):
		if torch.cuda.is_available():
			self.mask = self.mask.cuda()
		return Variable(self.mask, requires_grad=False)

	def forward(self, x):
		if self.mask_flag == True:
			mask_var = self.get_mask()
			weight = self.weight * mask_var
			return F.linear(x, weight, self.bias)
		else:
			return F.linear(x, self.weight, self.bias)



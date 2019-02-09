import torch
from torch.utils.data import Sampler
from torch.optim.optimizer import Optimizer
import numpy as np


class DataSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)



"""
class AdaptiveLR(object):
	# --- Reduce LR when cost curve flattens --- 
   
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
        
       
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_acc, val_loss = validate(...)
        >>>     scheduler.step(val_loss, epoch)

    def __init__(self, optimizer, mode='min', factor=0.1, patience=50,
                 verbose=0, epsilon=1e-4, cooldown=0, min_lr=0):
        super(ReduceLROnPlateau, self).__init__()

        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        
        if self.mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min' :
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def reset(self):
        self._reset()

    def step(self, metrics, epoch):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.lr_epsilon:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0

"""

class MyAdaptiveLR(object):
	# --- Reduce LR when cost curve flattens --- 

	def __init__(self, optimizer, factor=0.1, patience=50, stag_range=1e-4, cooldown=0, min_lr=0):
		
		super(MyAdaptiveLR, self).__init__()

		self.optimizer = optimizer  # Imported

		self.factor = factor  # Factor should be < 1.0
		self.min_lr = min_lr
		self.stag_range = stag_range  # Stagnation Range
		self.stag_count = 0  # Number of epochs that have stagnated
		self.patience = patience
		self.cooldown = cooldown  # Number of epochs to wait before resuming
		self.cooldown_counter = 0
		self.monitor_op = None
		self.lowest_val = np.Inf  


		self.optimizer = optimizer
		self._reset()



	def _reset(self):
		# -- Resets stagnation counter and cooldown counter -- 
		self.monitor_op = lambda a, b: np.less(a, b - self.stag_range)  # Tests whether it has decreased enough


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
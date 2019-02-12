import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from rbm import RBM


def pre_train_layer(dataloader,n_in,n_out,epochs,lr):

    rbm = RBM(n_in,n_out) 
    optimizer = torch.optim.SGD(rbm.parameters(),lr)

    for epoch in range(epochs):
        loss_ = []

        for batch in dataloader:
            data, _ = batch
            data = Variable(data.view(-1,n_in))
            sample_data = data.bernoulli()  # Takes in a range of 0-1

            v,v1 = rbm(sample_data)
            loss = rbm.free_energy(v) - rbm.free_energy(v1)
            
            loss_.append(loss.data[0])
            optimizer.zero_grad()  # Gradient needs to be reduced back to zero after each iteration
            loss.backward()
            optimizer.step()


        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_)))

    # -- Get weights from pretraining

    for name, param in rbm.named_parameters():
        if name in ['W']:
            return param


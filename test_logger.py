from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.trainer import Trainer
from   torch.utils.trainer.plugins.logger import Logger
from   torch.utils.trainer.plugins.progress import ProgressMonitor

class ShallowMLP(nn.Module):
    def __init__(self, shape, force_no_cuda=False):
        super(ShallowMLP, self).__init__()
        self.in_shape = shape[0]
        self.hidden_shape = shape[1]
        self.out_shape = shape[2]
        self.fc1 = nn.Linear(self.in_shape, self.hidden_shape)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_shape, self.out_shape)

        self.use_cuda = torch.cuda.is_available() and not force_no_cuda
        if self.use_cuda:
            self = self.cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return y

class SimpleDataset(object):
    def __init__(self, n, force_no_cuda=False):
        super(SimpleDataset, self)
        self.n = n
        self.i = 0
        self.use_cuda = torch.cuda.is_available() and not force_no_cuda

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()

    def next(self):
        if self.i >= self.n: 
            raise StopIteration()
        cur = self.i
        self.i += 1
        if self.use_cuda:
            return torch.cuda.FloatTensor([[cur]]), torch.cuda.FloatTensor([[cur]])
        else:
            return torch.FloatTensor([[cur]]), torch.FloatTensor([[cur]])

    def __len__(self):
        return self.n

if __name__=="__main__":
    force_no_cuda = True
    model = ShallowMLP((1,5,1), force_no_cuda=force_no_cuda)
    dataset = SimpleDataset(5, force_no_cuda)

    optimizer = optim.SGD(model.parameters(), 0.001)
    criterion = nn.L1Loss()
    train = Trainer(model, 
        criterion=criterion,
        optimizer=optimizer, 
        dataset=dataset)
    
    progress_plug = ProgressMonitor()
    logger_plug = Logger(["progress"], [(2, 'iteration')])

    train.register_plugin(progress_plug)
    train.register_plugin(logger_plug)
    train.run()

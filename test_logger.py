from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from   trainer import Trainer
from   trainer.plugins.logger import Logger
from   trainer.plugins.visdom_logger import *
from   trainer.plugins.progress import ProgressMonitor
from   trainer.plugins.random import RandomMonitor

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
    random_plug = RandomMonitor(10000)


    logger = Logger(["progress"], [(2, 'iteration')])
    text_logger = VisdomTextLogger(["progress"], [(2, 'iteration')], update_type='APPEND')
    scatter_logger = VisdomPlotLogger('SCATTER', ["progress.samples_used", "progress.percent"], [(2, 'iteration')])
    hist_logger = VisdomLogger(
        ["random.data"], 
        [(2, 'iteration')],
        opts=dict(numbins=20))

# viz.histogram(X=np.random.rand(10000), opts=dict(numbins=20))

    train.register_plugin(progress_plug)
    train.register_plugin(random_plug)

    train.register_plugin(logger)
    train.register_plugin(text_logger)
    train.register_plugin(scatter_logger)
    train.register_plugin(hist_logger)
    train.run()

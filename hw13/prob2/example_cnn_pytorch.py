import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import IPython

class CNN_PyTorch(nn.Module):
    def __init__(self, num_outputs):
        super(CNN_PyTorch, self).__init__()
        self.conv = nn.Conv2d(3, 80, 3, padding=1)
        self.pool = nn.MaxPool2d(5)
        self.lin = nn.Linear(25920, num_outputs) #25920 is gotten from arithmetic: after pool, variable is 18x18x80

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.lin(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
class CNN(object):

    def __init__(self,classes,image_size):
        '''
        Initializes the size of the network
        '''

        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = image_size

        self.output_size = self.num_class
        self.batch_size = 40

        self.net = self.build_network(num_outputs=self.output_size)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def build_network(self, num_outputs):
        return CNN_PyTorch(num_outputs)

    def parameters(self):
        return self.net.parameters()

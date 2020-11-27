import torch
import torch.nn as nn #neural network functionality
import torch.nn.functional as F #things for neural nets that are just functions, not objects
import numpy as np

import IPython

class CNN_PyTorch(nn.Module):
    def __init__(self, num_outputs):
        super(CNN_PyTorch, self).__init__()
        '''
        TODO: Initialize your layers here. Use nn as imported above.
        Notes: Make sure the convolution uses "same" padding; for a filter of size 2n+1, this means padding n.
        Everything with parameters needs to be declared here; other things can but don't have to be.
        Look at example_cnn_pytorch
        '''

    def forward(self, x):
        '''
        TODO: Compute your forward pass here, using the layers you initialized above.
        Notes: use view() to flatten your Variable after the convolution. num_flat_features below will be useful.
        Remember to put in relus where needed. 
        Look at example_cnn_pytorch
        '''

    def response_map(self, x):
        '''
        TODO: This should be basically one line: return the response Variable from the conv
        '''

    def num_flat_features(self, x): #this computes the number of elements in a multidimensional variable (ignoring batch)
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

    def get_acc(self,y_,y_out):

        '''
        Fill in a way to compute accuracy given the output of the network and the correct indices
        y_ (the true label) and y_out (the predict label)
        You'll want to convert from PyTorch Variables back to numpy (and maybe to a single scalar) here.
        '''
        _, predicted = torch.max(y_out, 1)
        total = len(y_)
        correct = (predicted == y_).sum().data.numpy()
        accuracy = correct / total
        return accuracy

    def accuracy(self, images, y_):
        return self.get_acc(y_, self.net(images))
    def parameters(self):
        return self.net.parameters()
    def response_map(self, x):
        return self.net.response_map(x)

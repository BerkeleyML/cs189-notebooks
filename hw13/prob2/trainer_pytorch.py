import datetime
import os
import sys
import argparse
import torch
from torch import Tensor, LongTensor
from torch.autograd import Variable

import IPython
import pickle


class Solver(object):

    def __init__(self, net, data):

     
        self.net = net
        self.data = data
       
        #Number of iterations to train for
        self.max_iter = 5000
        #Every this many iterations, record accuracy
        self.summary_iter = 200
        



        '''
        We'll use Stochastic Gradient Descent with momentum
        In the function optimize you will iteratively apply this on batches of data
        '''
        
        self.optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    def optimize(self):
        #record both of these accuracies every self.summary_iter iterations
        self.train_accuracy = []
        self.test_accuracy = []

        '''
        Performs the training of the network. 
        Implement SGD using the data manager to compute the batches
        Make sure to record the training and test accuracy through out the process
        Get batches of data from self.data, then take those numpy arrays and wrap them in Tensors wrapped in Variables
        Since pytorch expects indices, labels should be LongTensors within Variables.
        You can use self.net.accuracy, self.net.loss_fn, etc
        Note that self.net.net, which is an instance of nn.Module, can be called directly on data to produce output,
            which can then be used to compute loss.
        Variable.backward() does backprop
        '''

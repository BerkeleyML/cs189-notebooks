import IPython
from numpy.random import uniform
import random
import time

import numpy as np
import glob
import os

import matplotlib.pyplot as plt


import sys

from  sklearn.neighbors import KNeighborsClassifier



class NN(): 


	def __init__(self,train_data,val_data,n_neighbors=5):

		self.train_data = train_data
		self.val_data = val_data

		self.sample_size = 400

		self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

		
	def train_model(self): 

		'''
		Train Nearest Neighbors model
		'''



	def get_validation_error(self):

		'''
		Compute validation error. Please only compute the error on the sample_size number 
		over randomly selected data points. To save computation. 

		'''



	def get_train_error(self):

		'''
		Compute train error. Please only compute the error on the sample_size number 
		over randomly selected data points. To save computation. 
		'''



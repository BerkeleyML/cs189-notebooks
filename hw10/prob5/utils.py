
from numpy.random import uniform
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

from sklearn.preprocessing import StandardScaler




def create_one_hot_label(Y,N_C):
	''''
	Input
	Y: list of class labels (int)
	N_C: Number of Classes

	Returns
	List of one hot arrays with dimension N_C

	'''

	y_one_hot = []
	for y in Y:

		one_hot_label = np.zeros(N_C)

		one_hot_label[y] = 1.0
		y_one_hot.append(one_hot_label)

	return  y_one_hot


def subtract_mean_from_data(X,Y):
	''''
	Input
	X: List of data points
	Y: list of one hot class labels

	Returns
	X and Y with mean subtracted

	'''
	Y = np.array(Y)
	if len(Y.shape)==1:
		Y = np.reshape(Y, (-1, 1))

	ss_x = StandardScaler(with_std = False)
	ss_y = StandardScaler(with_std = False)

	ss_x.fit(X)
	X = ss_x.transform(X)

	ss_y.fit(Y)
	Y = ss_y.transform(Y)

	return X,Y

def compute_covariance_matrix(X,Y):
	''''
	Input
	X: List of data points
	Y: list of one hot class labels

	Returns
	Covariance Matrix of X and Y
	Note: Assumes Mean is subtracted

	'''

	dim_x = np.max(X[0].shape)
	dim_y = np.max(Y[0].shape)

	N = len(X)
	X = np.array(X)
	Y = np.array(Y)
	C_XY = X.T @ Y

	return C_XY/float(N)

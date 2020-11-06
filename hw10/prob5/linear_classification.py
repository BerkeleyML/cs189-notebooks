from numpy.random import uniform
import random
import time

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys

from projection import Project2D, Projections
from confusion_mat import getConfusionMatrixPlot

from ridge_model import Ridge_Model
from svm_model import SVM_Model
from logistic_model import Logistic_Model

CLASS_LABELS = ['apple', 'banana', 'eggplant']


class Model():
    """ Generic wrapper for specific model instance. """

    def __init__(self, model):
        """ Store specific pre-initialized model instance. """

        self.model = model

    def train_model(self, X, Y):
        """ Train using specific model's training function. """

        self.model.train_model(X, Y)

    def test_model(self, X, Y):
        """ Test using specific model's eval function. """
        if hasattr(self.model, "evals"):
            labels = np.array(Y)
            p_labels = self.model.evals(X)

        else:
            labels = []  # List of actual labels
            p_labels = []  # List of model's predictions
            success = 0  # Number of correct predictions
            total_count = 0  # Number of images

            for i in range(len(X)):

                x = X[i]  # Test input
                y = Y[i]  # Actual label
                y_ = self.model.eval(x)  # Model's prediction
                labels.append(y)
                p_labels.append(y_)

                if y == y_:
                    success += 1
                total_count += 1

        print("Computing Confusion Matrix")
        # Compute Confusion Matrix
        getConfusionMatrixPlot(labels, p_labels, CLASS_LABELS)


if __name__ == "__main__":
    # Load Training Data and Labels
    X = list(np.load('little_x_train.npy'))
    Y = list(np.load('little_y_train.npy'))

    # Load Validation Data and Labels
    X_val = list(np.load('little_x_val.npy'))
    Y_val = list(np.load('little_y_val.npy'))

    CLASS_LABELS = ['apple', 'banana', 'eggplant']

    # Project Data to 200 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim, CLASS_LABELS)
    cca_proj, white_cov = projections.cca_projection(X, Y, k=2)

    X = projections.project(cca_proj, white_cov, X)
    X_val = projections.project(cca_proj, white_cov, X_val)


    # ####RUN RIDGE REGRESSION#####
    # ridge_m = Ridge_Model(CLASS_LABELS)
    # model = Model(ridge_m)
    #
    # model.train_model(X, Y)
    # model.test_model(X, Y)
    # model.test_model(X_val, Y_val)
    #
    ####RUN LDA REGRESSION#####

    lda_m = LDA_Model(CLASS_LABELS)
    model = Model(lda_m)

    model.train_model(X, Y)
    model.test_model(X, Y)
    model.test_model(X_val, Y_val)

    # ####RUN QDA REGRESSION#####
    #
    # qda_m = QDA_Model(CLASS_LABELS)
    # model = Model(qda_m)
    #
    # model.train_model(X, Y)
    # model.test_model(X, Y)
    # model.test_model(X_val, Y_val)

    # ####RUN SVM REGRESSION#####
    #
    # svm_m = SVM_Model(CLASS_LABELS)
    # model = Model(svm_m)
    #
    # model.train_model(X, Y)
    # model.test_model(X, Y)
    # model.test_model(X_val, Y_val)
    #
    # ####RUN Logistic REGRESSION#####
    # lr_m = Logistic_Model(CLASS_LABELS)
    # model = Model(lr_m)
    #
    # model.train_model(X, Y)
    # model.test_model(X, Y)
    # model.test_model(X_val, Y_val)

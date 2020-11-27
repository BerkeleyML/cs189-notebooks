from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import cv2
import IPython
import numpy as np

from torch import Tensor
from torch.autograd import Variable

class Confusion_Matrix(object):


    def __init__(self,val_data,train_data, class_labels):

        self.val_data = val_data
        self.train_data = train_data
        self.CLASS_LABELS = class_labels
        #self.sess = sess


    def test_net(self, net):

        true_labels = []
        predicted_labels = []
        for datum in self.val_data:

            batch_eval = np.zeros([1,datum['features'].shape[0],datum['features'].shape[1],datum['features'].shape[2]])
            batch_eval[0,:,:,:] = datum['features']

            batch_label = np.zeros([1,len(self.CLASS_LABELS)])
            batch_label[0,:] = datum['label']
            prediction = net.net(Variable(Tensor(batch_eval))).data.numpy()

            class_pred = np.argmax(prediction)
            class_truth = datum['label']

            true_labels.append(class_truth)
            predicted_labels.append(class_pred)

        
        self.getConfusionMatrixPlot(true_labels,predicted_labels,self.CLASS_LABELS)


    def getConfusionMatrix(self,true_labels, predicted_labels):
        """
        Input
        true_labels: actual labels
        predicted_labels: model's predicted labels

        Output
        cm: confusion matrix (true labels vs. predicted labels)
        """

        # Generate confusion matrix using sklearn.metrics
        cm = confusion_matrix(true_labels, predicted_labels)
        return cm


    def plotConfusionMatrix(self,cm, alphabet):
        """
        Input
        cm: confusion matrix (true labels vs. predicted labels)
        alphabet: names of class labels

        Output
        Plot confusion matrix (true labels vs. predicted labels)
        """

        fig = plt.figure()
        plt.clf()                       # Clear plot
        ax = fig.add_subplot(111)       # Add 1x1 grid, first subplot
        ax.set_aspect(1)
        res = ax.imshow(cm, cmap=plt.cm.binary,
                        interpolation='nearest', vmin=0, vmax=80)

        plt.colorbar(res)               # Add color bar

        width = len(cm)                 # Width of confusion matrix
        height = len(cm[0])             # Height of confusion matrix

        # Annotate confusion entry with numeric value
        for x in range(width):
            for y in range(height):
                ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                            verticalalignment='center', color=self.getFontColor(cm[x][y]))


        # Plot confusion matrix (true labels vs. predicted labels)
        plt.xticks(range(width), alphabet[:width], rotation=90)
        plt.yticks(range(height), alphabet[:height])
        plt.show()
        return plt


    def getConfusionMatrixPlot(self,true_labels, predicted_labels, alphabet):
        """
        Input
        true_labels: actual labels
        predicted_labels: model's predicted labels
        alphabet: names of class labels

        Output
        Plot confusion matrix (true labels vs. predicted labels)
        """

        # Generate confusion matrix using sklearn.metrics
        cm = confusion_matrix(true_labels, predicted_labels)

        # Plot confusion matrix (true labels vs. predicted labels)
        return self.plotConfusionMatrix(cm, alphabet)


    def getFontColor(self,value):
        """
        Input
        value: confusion entry value

        Output
        font color for confusion entry
        """
        if value < -1:
            return "black"
        else:
            return "white"

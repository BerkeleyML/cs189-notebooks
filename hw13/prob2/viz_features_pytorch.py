from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import cv2
import IPython
import numpy as np

from torch import Tensor
from torch.autograd import Variable

class Viz_Feat(object):


    def __init__(self,val_data,train_data, class_labels):

        self.val_data = val_data
        self.train_data = train_data
        self.CLASS_LABELS = class_labels




    def vizualize_features(self,net):

        images = [0,10,100]
        '''
        Compute (and save, probably) the 5 filter responses for each of the 3 images in val data with indices given above
        Use revert_image to go from response back to an actual viewable / saveable image
        '''



    def revert_image(self,img):
        '''
        Used to revert images back to a form that can be easily visualized
        '''

        img = img/np.max(img)
        img = (img+1.0)/2.0*255.0

        img = np.array(img,dtype=int)

        blank_img = np.zeros([img.shape[0],img.shape[1],3])

        blank_img[:,:,0] = img
        blank_img[:,:,1] = img
        blank_img[:,:,2] = img

        img = blank_img.astype("uint8")

        return img

        





import os
import numpy as np
from numpy.random import random
import cv2
import copy
import glob

import pickle
import IPython

FLIPAUG = False


class data_manager(object):
    def __init__(self,classes,image_size,compute_features = None, compute_label = None):

        #Batch Size for training
        self.batch_size = 40
        #Batch size for test, more samples to increase accuracy (we use somewhat fewer here than tf uses for memory reasons)
        self.val_batch_size = 200

        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = image_size


        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))


        self.cursor = 0
        self.t_cursor = 0
        self.epoch = 1

        self.recent_batch = []

        if compute_features == None:
            self.compute_feature = self.compute_features_baseline
        else:
            self.compute_feature = compute_features

        if compute_label == None:
            self.compute_label = self.compute_label_baseline
        else:
            self.compute_label = compute_label


        self.load_train_set()
        self.load_validation_set()



    def get_train_batch(self):

        '''
        TODO: Compute a training batch for the neural network
        The batch size should be size 40 (self.batch_size)

        '''
        #FILL IN

 
    def get_empty_state(self): #get an empty np array in the correct shape to be a batch of images
        images = np.zeros((self.batch_size, 3, self.image_size,self.image_size))
        return images

    def get_empty_label(self): #get an empty np array in the correct shape to be a batch of labels
        return [0]*self.batch_size # in pytorch, since nothing is one-hot, labels is a 1d batch

    def get_empty_state_val(self):
        images = np.zeros((self.val_batch_size, 3, self.image_size,self.image_size))
        return images

    def get_empty_label_val(self):
        return [0]*self.val_batch_size # in pytorch, since nothing is one-hot, labels is a 1d batch



    def get_validation_batch(self):

        '''
        TODO: Compute a training batch for the neural network
        The batch size should be size 200 (self.val_batch_size)

        '''
        #FILL IN
        


    def compute_features_baseline(self, image):
        '''
        computes the featurized on the images. In this case this corresponds
        to rescaling and standardizing.
        '''

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image / 255.0) * 2.0 - 1.0

        image = np.array([image[:,:,0], image[:,:,1], image[:,:,2]]) #pytorch expects images in channel-major order
        return image


    def compute_label_baseline(self,label):
        '''
        Given class label, return index (pytorch doesn't use one-hot)
        '''

        idx = self.classes.index(label)

        return idx


    def load_set(self,set_name):

        '''
        Given a string which is either 'val' or 'train', the function should load all the
        data into an
        '''

        data = []
        data_paths = glob.glob(set_name+'/*.png')

        count = 0


        for datum_path in data_paths:

            label_idx = datum_path.find('_')


            label = datum_path[len(set_name)+1:label_idx]

            if self.classes.count(label) > 0:

                img = cv2.imread(datum_path)
                
                if FLIPAUG and set_name == 'train':
                    # random horizontal flip for training phase
                    if np.random.uniform() >0.5:
                        img = cv2.flip(img, 1)
                
                label_vec = self.compute_label(label)

                features = self.compute_feature(img)


                data.append({'c_img': img, 'label': label_vec, 'features': features})

        np.random.shuffle(data)
        return data


    def load_train_set(self):
        '''
        Loads the train set
        '''

        self.train_data = self.load_set('train')


    def load_validation_set(self):
        '''
        Loads the validation set
        '''

        self.val_data = self.load_set('val')

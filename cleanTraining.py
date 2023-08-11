import os
import random
import sys
import h5py
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from PIL import Image
import math
import pickle

from importlib.machinery import SourceFileLoader

# Add model name here
modelName = "cnn6"

# Global Variables
CLEAN_MODEL_FILEPATH = "/home/bt3/19CS10049/backdoorPytorch/Models/CleanModel{}.pt".format(modelName)
DATA_DIR = '/home/bt3/19CS10049/backdoorPytorch/GTSRB_data/'  # data folder

IMG_SHAPE = (32, 32, 3)
BATCH_SIZE = 128
device = "cuda"

def load_dataset_util(data_filename):
    ''' assume all datasets are numpy arrays '''
    dataset = {}

    if os.path.exists("dataPickle"):
        dbfile = open('dataPickle', 'rb')     
        dataset = pickle.load(dbfile)
        dbfile.close()
        return dataset

    trainDF = pd.read_csv(data_filename+"Train.csv", delimiter=',')
    testDF = pd.read_csv(data_filename+"Test.csv", delimiter=',')

    X_train = []
    Y_train = []
    for i in tqdm(range(trainDF.shape[0])):
        img = Image.open(data_filename+trainDF["Path"][i])
        img = img.resize((IMG_SHAPE[0], IMG_SHAPE[1]))
        img = np.array(img).astype('float32')
        img /= 255.0
        img -= 0.5
        X_train.append(img)
        Y_train.append(trainDF['ClassId'][i])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_test = []
    Y_test = []
    for i in tqdm(range(testDF.shape[0])):
        img = Image.open(data_filename+testDF["Path"][i])
        img = img.resize((IMG_SHAPE[0], IMG_SHAPE[1]))
        img = np.array(img).astype('float32')
        img /= 255.0
        X_test.append(img)
        Y_test.append(testDF['ClassId'][i])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    dataset = {
        "X_train" : X_train,
        "Y_train" : Y_train,
        "X_test" : X_test,
        "Y_test" : Y_test
    }

    dbfile = open('dataPickle', 'ab')
    pickle.dump(dataset, dbfile)                     
    dbfile.close()

    return dataset

def load_dataset(data_file=DATA_DIR):

    dataset = load_dataset_util(data_file)

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    return X_train, Y_train, X_test, Y_test



class DataGenerator():
    def __init__(self, X, Y, is_test=0):              
        self.X = X
        self.Y = Y
        self.is_test = is_test
        self.idx = 0
        self.indexes = np.arange(len(self.Y))
    
    def on_epoch(self):
        self.idx = 0
        if(self.is_test==0):
            np.random.shuffle(self.indexes)

    def gen_data(self):
        batch_X, batch_Y = [], []
        while 1:
            cur_idx = self.indexes[self.idx]

            self.idx += 1
            
            cur_x = self.X[cur_idx]
            cur_y = self.Y[cur_idx]
            batch_X.append(cur_x)
            batch_Y.append(cur_y)

            if len(batch_Y) == BATCH_SIZE:
                batch_X = torch.from_numpy(np.array(batch_X))
                batch_Y = torch.from_numpy(np.array(batch_Y))
                return batch_X.float(), batch_Y.long()

            elif self.idx==len(self.Y):
                return (torch.from_numpy(np.array(batch_X)).float(), torch.from_numpy(np.array(batch_Y)).long())

spec = SourceFileLoader("CNN6","./Models/CNN6.py").load_module()

def evalModel():
    train_X, train_Y, test_X, test_Y = load_dataset()                       # Load training and testing data
    print("Data Loaded")
    model = spec.CNN6().float().to(device)                                       # Build a CNN model
    print("Model Loaded")

    train_gen = DataGenerator(train_X, train_Y)
    test_gen = DataGenerator(test_X, test_Y, 1)
    print("Data Segregated")
    print("Starting Model Training")

    
    loss = nn.CrossEntropyLoss()
    number_images = len(train_Y)
    numImages = len(test_Y)
    stps = int(numImages//BATCH_SIZE)
    model.fit(train_gen, epochs = 200, verbose = 1, steps_per_epoch = int(number_images//BATCH_SIZE), learning_rate = 0.0008, loss = loss, change_lr_every = 100, test_gen = test_gen, stps = stps, model_path = CLEAN_MODEL_FILEPATH)

    print("Evaluating model")
    model.load_state_dict(torch.load(CLEAN_MODEL_FILEPATH))
    number_images = len(test_Y)
    steps_per_epoch = int(number_images//BATCH_SIZE)
    acc, _ = model.evaluate(test_gen, steps_per_epoch, loss, 1)
    print('Final Test Accuracy: {:.4f}'.format(acc))

if __name__ == '__main__':
    evalModel()
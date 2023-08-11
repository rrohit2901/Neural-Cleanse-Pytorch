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
from torchvision import transforms

from tqdm import tqdm
from PIL import Image
import math
import pickle

from importlib.machinery import SourceFileLoader

# Add model name here
modelName = "cnn6"

# Global Variables
MODEL_FILEPATH = "/home/bt3/19CS10049/backdoorPytorch/Models/InfectedModel{}.pt".format(modelName)
INFECTED_MODEL_FILEPATH = "/home/bt3/19CS10049/backdoorPytorch/Models/CleanModel{}.pt".format(modelName)
DATA_DIR = '/home/bt3/19CS10049/backdoorPytorch/GTSRB_data/'  # data folder

# These are paramters for injecting backdoor
TARGET_LS = [28]
NUM_LABEL = len(TARGET_LS)
NUM_CLASSES = 43
PER_LABEL_RARIO = 0.1
INJECT_RATIO = (PER_LABEL_RARIO * NUM_LABEL) / (PER_LABEL_RARIO * NUM_LABEL + 1)
NUMBER_IMAGES_RATIO = 1 / (1 - INJECT_RATIO)
PATTERN_PER_LABEL = 1

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

def make_mask_pattern(image_row, image_col, channel_num, margin, pattern_size):   # This function creates a white square on bottom right pattern
    mask = np.zeros((image_row, image_col, channel_num))
    pattern = np.zeros((image_row, image_col, channel_num))
    mask[image_row - margin - pattern_size:image_row - margin, image_col - margin - pattern_size:image_col - margin,:] = 1
    pattern[image_row - margin - pattern_size:image_row - margin, image_col - margin - pattern_size:image_col - margin, :] = 1.

    return mask, pattern

def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img

def infect_X(img, tgt):
    mask, pattern = make_mask_pattern(img.shape[0], img.shape[1], img.shape[2], 1, 3)                       
    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)

    adv_img = injection_func(mask, pattern, adv_img)

    return adv_img, tgt



class DataGenerator():
    def __init__(self, target_ls, X, Y, inject_ratio, is_test=0):                       # target_ls is list of all possible targets (constrained to length 1 in this implementation)
        self.target_ls = target_ls
        self.X = X
        self.Y = Y
        self.inject_ratio = inject_ratio
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
            inject_ptr = random.uniform(0, 1)
            cur_idx = self.indexes[self.idx]

            self.idx += 1
            
            cur_x = self.X[cur_idx]
            cur_y = self.Y[cur_idx]
            
            if inject_ptr < self.inject_ratio:
                tgt = random.choice(self.target_ls)
                cur_x, _ = infect_X(cur_x, tgt)

            batch_X.append(cur_x)
            batch_Y.append(cur_y)

            if len(batch_Y) == BATCH_SIZE:
                batch_X = torch.from_numpy(np.array(batch_X))
                batch_Y = torch.from_numpy(np.array(batch_Y))
                return batch_X.float(), batch_Y.long()

            elif self.idx==len(self.Y):
                return (torch.from_numpy(np.array(batch_X)).float(), torch.from_numpy(np.array(batch_Y)).long())

spec = SourceFileLoader("CNN6","Models/CNN6.py").load_module()

def inject_backdoor():
    train_X, train_Y, test_X, test_Y = load_dataset()                       # Load training and testing data
    print("Data Loaded")
    model = spec.CNN6().float().to(device)                                       # Build a CNN model
    model.load_state_dict(torch.load(INFECTED_MODEL_FILEPATH))
    print("Model Loaded")

    train_gen = DataGenerator(TARGET_LS, train_X, train_Y, INJECT_RATIO, 0)
    test_adv_gen = DataGenerator(TARGET_LS, test_X, test_Y, 1, 1)
    test_clean_gen = DataGenerator(TARGET_LS, test_X, test_Y, 0, 1)
    print("Data Segregated")
    print("Starting Model Training")

    loss = nn.CrossEntropyLoss()
    number_images = len(train_Y)
    lr = 0.0005
    model.fit(train_gen, epochs = 100, verbose = 1, steps_per_epoch = int(number_images//BATCH_SIZE), learning_rate = lr, loss = loss, change_lr_every = 35)

    print("Saving Infected Model")
    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)

    print("Evaluating model")
    number_images = len(test_Y)
    steps_per_epoch = int(number_images//BATCH_SIZE)
    acc, _ = model.evaluate(test_clean_gen, steps_per_epoch, loss, 1)
    backdoor_acc, _ = model.evaluate(test_adv_gen, steps_per_epoch, loss, 1)
    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))

if __name__ == '__main__':
    inject_backdoor()
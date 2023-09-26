import os
import random
import sys
import h5py
import random
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from tqdm import tqdm
from PIL import Image
import statistics

from importlib.machinery import SourceFileLoader

# Global Variables
# Add model name here
modelName = "cnn6"

# Global Variables
MODEL_FILEPATH = "/home/bt3/19CS10049/backdoorPytorch/Models/InfectedModel{}.pt".format(modelName)
DATA_DIR = '/home/bt3/19CS10049/backdoorPytorch/GTSRB_data/'  # data folder
NUM_CLASSES = 43
IMG_SHAPE = (32, 32, 3)
BATCH_SIZE = 32
device = "cuda"

def load_dataset_util(data_filename):
    ''' assume all datasets are numpy arrays '''
    dataset = {}

    if os.path.exists("testPickle"):
        dbfile = open('testPickle', 'rb')     
        dataset = pickle.load(dbfile)
        dbfile.close()
        return dataset

    testDF = pd.read_csv(data_filename+"Test.csv", delimiter=',')

    X_test = []
    for i in tqdm(range(testDF.shape[0])):
        img = Image.open(data_filename+testDF["Path"][i])
        img = img.resize((IMG_SHAPE[0], IMG_SHAPE[1]))
        X_test.append(np.array(img))
    X_test = np.array(X_test)
    Y_test = np.array(testDF['ClassId'])

    dataset = {
        "X_test" : X_test,
        "Y_test" : Y_test
    }

    dbfile = open('testPickle', 'ab')
    pickle.dump(dataset, dbfile)                     
    dbfile.close()

    return dataset

def load_dataset(data_file=DATA_DIR):

    dataset = load_dataset_util(data_file)
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    return X_test, Y_test

def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img

def infect_X(img, tgt, mask, pattern):                      
    raw_img = torch.clone(img)
    adv_img = torch.clone(raw_img)

    adv_img = injection_func(mask, pattern, adv_img)

    return adv_img, tgt

class DataGenerator():
    def __init__(self, target_ls, X, Y, is_test=0):                       # target_ls is list of all possible targets (constrained to length 1 in this implementation)
        self.target_ls = target_ls
        self.X = X
        self.Y = Y
        self.is_test = is_test
        self.idx = 0
        self.indexes = np.arange(len(self.Y))
        self.inject_ratio = 1
    
    def on_epoch(self):
        self.idx = 0
        if(self.is_test):
            np.random.shuffle(self.indexes)
    
    def num_steps(self):
        num = (int)(len(self.X)//BATCH_SIZE)
        return num

    def gen_data(self, mask, pattern):
        batch_X, batch_Y = [], []
        while self.idx<len(self.indexes):
            inject_ptr = random.uniform(0, 1)
            cur_idx = self.indexes[self.idx]

            self.idx += 1
            
            cur_x = torch.from_numpy(self.X[cur_idx])
            cur_y = self.Y[cur_idx]
            
            if inject_ptr < self.inject_ratio:
                tgt = self.target_ls
                cur_x, cur_y = infect_X(cur_x, tgt, mask, pattern)

            batch_X.append(cur_x)
            batch_Y.append(cur_y)

            if len(batch_Y) == BATCH_SIZE:
                batch_X = torch.stack(batch_X, dim = 0)
                batch_Y = torch.from_numpy(np.array(batch_Y))
                return batch_X.float(), batch_Y.long()

            elif self.idx==len(self.Y):
                return (torch.stack(batch_X, dim = 0).float(), torch.from_numpy(np.array(batch_Y)).long())

def mad_outlier(masks):
    med = statistics.median(masks)
    mini = 0
    mad = [abs(i-med) for i in masks]
    med1 = statistics.median(mad)
    tgt = None
    max_aqi = 0.0
    C = 1.4826
    for i,val in enumerate(masks):
        if val<masks[mini]:
            mini = i
    max_aqi = abs(masks[mini] - med)/(C * med1)
    tgt = mini
    return tgt, max_aqi

def lossFunc(criterion, out, actual, mask, pattern, weight):
    lossF = criterion(out, actual) + weight * torch.norm(mask)
    return lossF

def find_min_change(model, test_gen, epochs, steps_per_epoch, learning_rate, loss, init_weight, mask, pattern, verbose, early_stopping_patience, early_stopping_threshold):
    weight = 0
    weight_up_scale = 2
    weight_down_scale = weight_up_scale ** 1.5
    weight_up_count = 0
    weight_down_count = 0
    weight_set_counter = 0
    patience = 5

    best_mask = None
    best_pattern = None
    best_reg = float('inf')
    loss_threshold = 0.99

    early_stopping_counter = 0
    early_stopping_flag = True

    mask.requires_grad = True
    pattern.requires_grad = True
    optimizer = optim.Adam([mask, pattern], lr = learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    print("Running Model")
    print("Initial norm of mask - {}".format(torch.norm(mask)))
    for epoch in range(epochs):
        running_loss = 0.0
        test_gen.on_epoch()
        cnt = 0
        for step in range(steps_per_epoch):
            data_x, data_y = test_gen.gen_data(mask, pattern)
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            data_x = data_x.permute(0, 3, 1, 2) 
            optimizer.zero_grad()
            out = model.forward(data_x)
            loss = lossFunc(criterion, out, data_y, mask, pattern, weight)
            loss.backward()
            optimizer.step()

            cnt += len(data_x)
            running_loss += loss.item()

        test_gen.on_epoch()
        accuracy, lossTest = model.evaluate(test_gen, steps_per_epoch, criterion, verbose, mask, pattern)
        reg_loss = torch.norm(mask)

        if accuracy>=loss_threshold and reg_loss<best_reg:
            best_mask = mask.detach().clone().cpu()
            best_pattern = pattern.detach().clone().cpu()
            trigger = best_mask * best_pattern
            image = transforms.ToPILImage()(best_mask.permute(2,0,1)).convert("RGB")
            image.save("RTrigger/cnnMask.png")
            image = transforms.ToPILImage()(best_pattern.permute(2,0,1)).convert("RGB")
            image.save("RTrigger/cnnPattern.png")
            image = transforms.ToPILImage()(trigger.permute(2,0,1)).convert("RGB")
            image.save("RTrigger/cnnTrigger.png")
            best_reg = int(reg_loss)

        print(weight)
        if weight==0 and accuracy>=loss_threshold:
            weight_set_counter += 1
            if weight_set_counter >= patience:
                weight = init_weight
                weight_up_count = 0
                weight_down_count = 0
        else:
            weight_set_counter = 0
        
        if accuracy>=loss_threshold:
            weight_up_count += 1
            weight_down_count = 0
        else:
            weight_down_count += 1
            weight_up_count = 0
        
        if weight_up_count >= patience:
            weight = weight * weight_up_scale
            weight_up_count = 0
        elif weight_down_count >= patience:
            weight /= weight_down_scale
            weight_down_count = 0
        
        if early_stopping_flag:
            if accuracy<early_stopping_threshold:
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= early_stopping_patience:
                early_stopping_flag = False
                print("Early stopping")
                break

        if(verbose):
            print("Minimum mask size required = {}".format(torch.norm(mask)))
    return mask, pattern

def backdoor_identification():
    masks = []
    patterns = []

    weight = 0.001
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1

    early_stopping_patience = 10
    early_stopping_threshold = 1.0

    test_X, test_Y = load_dataset()

    spec = SourceFileLoader("CNN6","Models/CNN6.py").load_module()

    for target in range(NUM_CLASSES):
        print("Trying for Target - {}".format(target))
        model = spec.CNN6().float().to(device)                                       # Build a CNN model
        model.load_state_dict(torch.load(MODEL_FILEPATH))
        mask = torch.zeros(IMG_SHAPE)
        pattern = torch.zeros(IMG_SHAPE)
        test_gen = DataGenerator(target, test_X, test_Y, 0)
        mask, pattern = find_min_change(model, test_gen, 100, test_gen.num_steps(), learning_rate, criterion, weight, mask, pattern, 1, early_stopping_patience, early_stopping_threshold)
        masks.append(mask)
        patterns.append(pattern)
    
    masks_size = [torch.norm(i) for i in masks]
    target, aqi = mad_outlier(masks_size)

    print("Potential target can be {}, with AQI = {}".format(target, aqi))

    print("Saving identified trigger")
    trigger = masks[target] * patterns[target]
    image = transforms.ToPILImage()(trigger.permute(2,0,1)).convert("RGB")
    image.save("RTrigger/cnnTrigger.png")

if __name__ == '__main__':
    backdoor_identification()
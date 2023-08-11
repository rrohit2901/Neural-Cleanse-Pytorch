import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np

device = "cuda"

class CNN6(nn.Module):
    def __init__(self, base=32, dense=512, num_classes=43, input_shape=(32,32,3)):
        super().__init__()
        self.input_x = input_shape[0]
        self.input_y = input_shape[1]
        self.input_z = input_shape[2]

        self.conv1_1 = nn.Conv2d(self.input_z, base, kernel_size = 3, padding = "same")
        torch.nn.init.xavier_uniform_(self.conv1_1.weight)
        self.conv1_2 = nn.Conv2d(base, base, kernel_size = 3)
        torch.nn.init.xavier_uniform_(self.conv1_2.weight)
        self.relu1_1 = nn.ReLU()
        self.relu1_2 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2)
        self.dropout1 = nn.Dropout(p = 0.2)

        self.conv2_1 = nn.Conv2d(base, 2*base, kernel_size = 3, padding = "same")
        torch.nn.init.xavier_uniform_(self.conv2_1.weight)
        self.conv2_2 = nn.Conv2d(2*base, 2*base, kernel_size = 3)
        torch.nn.init.xavier_uniform_(self.conv2_2.weight)
        self.relu2_1 = nn.ReLU()
        self.relu2_2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2)
        self.dropout2 = nn.Dropout(p = 0.2)

        self.conv3_1 = nn.Conv2d(2*base, 4*base, kernel_size = 3, padding = "same")
        torch.nn.init.xavier_uniform_(self.conv3_1.weight)
        self.conv3_2 = nn.Conv2d(4*base, 4*base, kernel_size = 3)
        torch.nn.init.xavier_uniform_(self.conv3_2.weight)
        self.relu3_1 = nn.ReLU()
        self.relu3_2 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2)
        self.dropout3 = nn.Dropout(p = 0.2)

        self.fc1 = nn.Linear(512, dense)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(dense, num_classes)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.softmax = nn.Softmax(1)

    def forward(self, X):
        out1 = self.relu1_1(self.conv1_1(X))
        out1 = self.dropout1(self.max_pool1(self.relu1_2(self.conv1_2(out1))))
        
        out2 = self.relu2_1(self.conv2_1(out1))
        out2 = self.dropout2(self.max_pool2(self.relu2_2(self.conv2_2(out2))))

        out3 = self.relu3_1(self.conv3_1(out2))
        out3 = self.dropout3(self.max_pool3(self.relu3_2(self.conv3_2(out3))))

        out4 = torch.flatten(out3, start_dim=1)
        out4 = self.dropout4(self.relu4(self.fc1(out4)))
        out4 = self.softmax(self.fc2(out4))

        return out4
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        return
    
    def fit(self, train_gen, epochs, verbose, steps_per_epoch, learning_rate, loss, change_lr_every, test_gen = None, stps = None, model_path = None):
        optimizer = optim.Adam(self.parameters(), lr = learning_rate, weight_decay = 0.0001)
        bestAccuracy = None
        for epoch in range(epochs):
            if(epoch % change_lr_every == change_lr_every-1):
                learning_rate = learning_rate / 2
                optimizer = optim.Adam(self.parameters(), lr = learning_rate)
            train_gen.on_epoch()
            running_loss = 0.0
            y_pred = []
            y_act = []
            for step in range(steps_per_epoch):
                data_x, data_y = train_gen.gen_data()
                
                optimizer.zero_grad()
                data_x = data_x.to(device)
                data_y = data_y.to(device)
                data_x = data_x.permute(0, 3, 1, 2)
                out = self.forward(data_x)
                lossF = loss(out, data_y)
                lossF.backward()
                optimizer.step()
                running_loss += lossF.item()
                y_pred.append(torch.argmax(out, dim = 1).cpu().numpy())
                y_act.append(data_y.cpu().numpy())
                
            y_pred = np.array(y_pred).flatten()
            y_act = np.array(y_act).flatten()

            Accuracy = (sum([y_pred[i]==y_act[i] for i in range(len(y_pred))])) / len(y_pred)
            if(verbose):
                print(running_loss, steps_per_epoch)
                print("Epoch -- {} ; Average Loss -- {} ; Accuracy -- {}".format(epoch, running_loss/(steps_per_epoch), Accuracy))
            if(test_gen != None):
                accuracy, _  = self.evaluate(test_gen, stps, loss, verbose)
                if(bestAccuracy == None or accuracy > bestAccuracy):
                    bestAccuracy = accuracy
                    self.save(model_path)
        print("Training Done")
        return

    def evaluate(self, test_gen, steps_per_epoch, loss, verbose, mask = None, pattern = None):
        running_loss = 0.0
        y_pred = []
        y_act = []

        test_gen.on_epoch()
        self.eval()
        
        for step in range(steps_per_epoch):
            with torch.no_grad():
                if mask is not None:
                    data_x, data_y = test_gen.gen_data(mask = mask, pattern = pattern)
                else:
                    data_x, data_y = test_gen.gen_data()
                data_x = data_x.to(device)
                data_y = data_y.to(device)
                data_x = data_x.permute(0, 3, 1, 2)
                out = self.forward(data_x)

                y_pred.append(torch.argmax(out, dim = 1).cpu().numpy())
                y_act.append(data_y.cpu().numpy())

                lossF = loss(out, data_y)
                
                running_loss += lossF.item()
                
                
        
        y_pred = np.array(y_pred).flatten()
        y_act = np.array(y_act).flatten()

        Accuracy = (sum([y_pred[i]==y_act[i] for i in range(len(y_pred))])) / len(y_pred)
        running_loss /= steps_per_epoch

        if(verbose):
            print("Accuracy on provided Data -- {} ; Loss -- {}".format(Accuracy, running_loss))
        

        return Accuracy, running_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import timm

device = "cuda"

class deit(nn.Module):
    def __init__(self, num_classes = 43):
        super(deit, self).__init__()
        self.model = timm.create_model("deit3_small_patch16_224", pretrained=True)
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.linear = nn.Linear(1000, num_classes)

        self.softmax = nn.Softmax(dim = 0)

    def forward(self, x):
        # x = self.preprocess(x)
        out1 = self.linear(self.model(x))
        return self.softmax(out1)

    def save(self, path):
        torch.save(self.state_dict(), path)
        return
    
    def fit(self, train_gen, epochs, verbose, steps_per_epoch, learning_rate, loss, change_lr_every, test_gen = None, stps = None, model_path = None):
        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        bestAccuracy = 0
        for epoch in range(epochs):
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
                print("Epoch -- {} ; Average Loss -- {} ; Accuracy -- {}".format(epoch, running_loss/(steps_per_epoch), Accuracy))
            if(test_gen != None):
                Accuracy, loss = self.evaluate(test_gen, stps, loss, verbose)
                if(Accuracy > bestAccuracy):
                    bestAccuracy = Accuracy
                    self.save(model_path)
        print("Training Done")
        return

    def evaluate(self, test_gen, steps_per_epoch, loss, verbose):
        running_loss = 0.0
        y_pred = []
        y_act = []

        test_gen.on_epoch()
        self.eval()
        for step in range(steps_per_epoch):
            with torch.no_grad():
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
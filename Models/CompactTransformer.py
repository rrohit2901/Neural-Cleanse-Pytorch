from typing import Callable
import time
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from PIL import Image
from einops import rearrange, reduce, repeat
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.optim as optim

device = "cuda"

class MSA(nn.Module):
    """Multi-head Self Attention Block"""

    def __init__(
        self, heads: int, emb_dim: int, 
        dropout: float = 0., attention_dropout: float = 0.
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_h = heads
        self.head_dim = self.emb_dim // self.n_h
        self.q = nn.Linear(self.emb_dim, self.emb_dim)
        self.k = nn.Linear(self.emb_dim, self.emb_dim)
        self.v = nn.Linear(self.emb_dim, self.emb_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.linear_projection = nn.Linear(self.emb_dim, self.emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # (bs,     s_l,      e_d)
        batch_s, seq_len, emb_dim = x.shape
        # (bs, s_l, e_d) -> (bs, s_l, n_h, h_d) -> (bs, n_h, s_l, h_d)
        x_q = self.q(x).view(
            batch_s, seq_len, self.n_h, self.head_dim).transpose(1, 2)
        x_k = self.k(x).view(
            batch_s, seq_len, self.n_h, self.head_dim).transpose(1, 2)
        x_v = self.v(x).view(
            batch_s, seq_len, self.n_h, self.head_dim).transpose(1, 2)
        # @ operator is the convention for matrix multiplication, throughout python
        # q @ k.T -> (bs, n_h, s_l, h_d) @ (bs, n_h, h_d, s_l) -> (bs, n_h, s_l, s_l)
        # Softmax((q @ k.T)/root(h_d)) @ v
        #   -> (bs, n_h, s_l, s_l) @ (bs, n_h, s_l, h_d) -> (bs, n_h, s_l, h_d)
        attention = (x_q @ x_k.transpose(-2, -1)) / math.sqrt(x_q.size(-1))
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        # (bs, n_h, s_l, h_d) -> (bs, s_l, n_h, h_d) -> (bs, s_l, e_d)
        x = (attention @ x_v).transpose(1, 2).reshape(batch_s, seq_len, emb_dim)
        x = self.linear_projection(x)
        x = self.dropout(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block"""

    def __init__(
        self, n_h: int, emb_dim: int, feat_dim: int, 
        dropout: float = 0, attention_dropout: float = 0
    ):
        super().__init__()
        self.msa = MSA(heads=n_h, emb_dim=emb_dim, dropout=dropout, attention_dropout=attention_dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = MLP(emb_dim, feat_dim, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor):
        identity = x
        x = self.msa(x)
        x += identity
        x = self.norm1(x)
        identity = x
        x = self.ffn(x)
        x += identity
        x = self.norm2(x)
        return x

class MLP(nn.Module):
    """MLP block"""

    def __init__(self, emb_dim: int, feat_dim: int, dropout: float = 0):
        super().__init__()
        self.layer1 = nn.Linear(emb_dim, feat_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(feat_dim, emb_dim)

        # below init from torchvision
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.normal_(self.layer1.bias, std=1e-6)
        nn.init.normal_(self.layer2.bias, std=1e-6)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x

class ConvTokenizer(nn.Module):
    def __init__(
        self,
        channels: int = 3, emb_dim: int = 256,
        conv_kernel: int = 3, conv_stride: int = 2, conv_pad: int = 3,
        pool_kernel: int = 3, pool_stride: int = 2, pool_pad: int = 1,
        activation: Callable = nn.ReLU()
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels, out_channels=emb_dim,
            kernel_size=conv_kernel, stride=conv_stride,
            padding=(conv_pad, conv_pad)
        )
        self.act = activation(inplace=True)
        self.max_pool = nn.MaxPool2d(
            kernel_size=pool_kernel, stride=pool_stride, 
            padding=pool_pad
        )
            
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        x = self.max_pool(x)
        return x

class SeqPool(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.dense = nn.Linear(emb_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, seq_len, emb_dim = x.shape
        identity = x
        x = self.dense(x)
        x = rearrange(
            x, 'bs seq_len 1 -> bs 1 seq_len', seq_len=seq_len
        )
        x = self.softmax(x)
        x = x @ identity
        x = rearrange(
            x, 'bs 1 e_d -> bs e_d', e_d=emb_dim
        )
        return x

class CCT(nn.Module):
    """
        Compact Convolutional Transformer (CCT) Model
        https://arxiv.org/abs/2104.05704v4
    """    
    def __init__(
        self,
        conv_kernel: int = 3, conv_stride: int = 2, conv_pad: int = 3,
        pool_kernel: int = 3, pool_stride: int = 2, pool_pad: int = 1,
        heads: int = 4, emb_dim: int = 256, feat_dim: int = 2*256, 
        dropout: float = 0.1, attention_dropout: float = 0.1, layers: int = 7, 
        channels: int = 3, image_size: int = 32, num_class: int = 43
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.image_size = image_size

        self.tokenizer = ConvTokenizer(
            channels=channels, emb_dim=self.emb_dim,
            conv_kernel=conv_kernel, conv_stride=conv_stride, conv_pad=conv_pad,
            pool_kernel=pool_kernel, pool_stride=pool_stride, pool_pad=pool_pad,
            activation=nn.ReLU
        )

        with torch.no_grad():
            x = torch.randn([1, channels, image_size, image_size])
            out = self.tokenizer(x)
            _, _, ph_c, pw_c  = out.shape

        self.linear_projection = nn.Linear(
            ph_c, pw_c, self.emb_dim
        )

        self.pos_emb = nn.Parameter(
            torch.randn(
                [1, ph_c*pw_c, self.emb_dim]
            ).normal_(std=0.02) # from torchvision, which takes this from BERT
        )
        self.dropout = nn.Dropout(dropout)
        encoders = []
        for _ in range(0, layers):
            encoders.append(
                TransformerEncoderBlock(
                    n_h=heads, emb_dim=self.emb_dim, feat_dim=feat_dim,
                    dropout=dropout, attention_dropout=attention_dropout
                )
            )
        self.encoder_stack = nn.Sequential(*encoders)
        self.seq_pool = SeqPool(emb_dim=self.emb_dim)
        self.mlp_head = nn.Linear(self.emb_dim, num_class)


    def forward(self, x: torch.Tensor):     
        bs, c, h, w = x.shape  # (bs, c, h, w)

        # Creates overlapping patches using ConvNet
        x = self.tokenizer(x)
        x = rearrange(
            x, 'bs e_d ph_h ph_w -> bs (ph_h ph_w) e_d', 
            bs=bs, e_d=self.emb_dim
        )

        # Add position embedding
        x = self.pos_emb.expand(bs, -1, -1) + x
        x = self.dropout(x)

        # Pass through Transformer Encoder layers
        x = self.encoder_stack(x)

        # Perform Sequential Pooling <- Novelty of the paper
        x = self.seq_pool(x)

        # MLP head used to get logits
        x = self.mlp_head(x)

        return x
    def save(self, path):
        torch.save(self.state_dict(), path)
        return
    
    def fit(self, train_gen, epochs, verbose, steps_per_epoch, learning_rate, loss, change_lr_every, test_gen = None, stps = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model_path = None):
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
                accuracy, _ = self.evaluate(test_gen, stps, loss, verbose)
                if(accuracy > bestAccuracy):
                    bestAccuracy = accuracy
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
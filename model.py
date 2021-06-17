import torch.nn as nn
import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50) 
        self.fc2 = nn.Linear(50,36)
    
    def forward(self,x):
        # print("initially: ",x.shape)
        x=self.conv1(x)
        # print("after conv1: ",x.shape)
        x=F.max_pool2d(x,2)
        # print("after maxpool2d: & relu",x.shape)
        x=F.relu(x)
        x= self.conv2(x)
        # print("after conv2: ",x.shape)
        x=F.max_pool2d(self.conv2_drop(x), 2)
        # print("after maxpool2d: & relu",x.shape)
        x = F.relu(x)
        x = x.view(-1, 320)
        # print("after view: ",x.shape)
        x=self.fc1(x)
        # print("after fc1: ",x.shape)
        x = F.relu(x)
        # print("after relu: ",x.shape)
        x = F.dropout(x)
        # print("after dropout: ",x.shape)
        x = self.fc2(x)
        # print("after fc2: ",x.shape)
        return x 
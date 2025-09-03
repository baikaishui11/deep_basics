import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR


class FullConnect(nn.Module):
    def __init__(self, k, n):
        super(FullConnect, self).__init__()
        self.full_connect1 = nn.Linear(30, 20)
        self.full_connect2 = nn.Linear(20, 10)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

    def forward(self, input):
        x = self.full_connect1(input)
        x = self.activation1(x)
        x = self.full_connect2(x)
        x = self.activation2(x)
        return x


def full_connect_demo():
    model = FullConnect(30, 10)
    input = torch.rand(4, 30)

    loss_fuction = nn.CrossEntropyLoss()
    label = torch.tensor([2, 3, 2, 4])
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    for i in range(10):
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fuction(output, label)

        loss.backward()
        # print(model.full_connect1.weight[0])
        optimizer.step()
        # print(model.full_connect1.weight[0])

        print(loss)
# def fuctional_demo():
#     a = torch.01tensor(torch.randn(2, 3))
#     x = torch.relu(a)
#     c = nn.ReLU()(a)
#     v = F.relu(a)

def container_demo():
    # model = nn.Sequential(nn.Conv2d(1, 20, 5, padding=2),
    #                       nn.ReLU(),
    #                       nn.Conv2d(20, 64, 5, padding=2),
    #                       nn.ReLU())
    # input = torch.rand(4, 1, 112, 112)
    # output = model(input)
    # print(output.shape)
    model = nn.ModuleList([nn.Conv2d(1, 20, 5, padding=2),
                          nn.ReLU(),
                          nn.Conv2d(20, 64, 5, padding=2),
                          nn.ReLU()])
    input = torch.rand(4, 1, 112, 112)
    for layer in model:
        input = layer(input)
    print(input.shape)

if __name__ == "__main__":
    # full_connect_demo()
    container_demo()
    print("run module_demo successful")
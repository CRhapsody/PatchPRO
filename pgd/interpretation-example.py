# construct two layer neural network with one hidden layer, which layer has two neurons
# and the output layer also has two neurons
# the input layer has two neurons
# the activation function is relu

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ExampleNet(nn.Module):
    def __init__(self,options = 'nnrepair'):
        super(ExampleNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.options = options
        if self.options == 'nnrepair':
            self.set_weights_nnrepair()
            self.set_radius_nnrepair()
        elif self.options == 'deeppoly':
            self.set_weights_deeppoly()
            self.set_radius_deeppoly()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    def set_weights_nnrepair(self):
        '''
        only use first two layers
        '''
        f1_weight = torch.tensor([[-1, 0.5], [2.0, 1]]).T
        f1_bias = torch.tensor([0., 0.])
        f2_weight = torch.tensor([[2, -0.5], [-1.5, 3]]).T
        f2_bias = torch.tensor([0., 0.])
        self.fc1.weight = torch.nn.Parameter(f1_weight)
        self.fc1.bias = torch.nn.Parameter(f1_bias)
        self.fc2.weight = torch.nn.Parameter(f2_weight)
        self.fc2.bias = torch.nn.Parameter(f2_bias)
    
    def set_weights_deeppoly(self):
        '''
        modified, original deeppoly example does not work
        '''
        f1_weight = torch.tensor([[1., 1.], [1., -1.]]).T
        f1_bias = torch.tensor([0., 0.])
        f2_weight = torch.tensor([[1., 1.], [1., -1.]]).T
        f2_bias = torch.tensor([0., 0.])
        self.fc1.weight = torch.nn.Parameter(f1_weight)
        self.fc1.bias = torch.nn.Parameter(f1_bias)
        self.fc2.weight = torch.nn.Parameter(f2_weight)
        self.fc2.bias = torch.nn.Parameter(f2_bias)
    # @property
    def inputs(self):
        if self.options == 'nnrepair':
            # [2,-1];[0.5,1];[0.5,4]
            return torch.tensor([[2.5,-1],[0.5,0.5],[0,4],]).to(self.fc1.weight.dtype)
        elif self.options == 'deeppoly':
            # [-0.5,-1],[2,1]
            # [-1,1],[1.5,1.5]
            return torch.tensor([[-1,1],[1.5,1.5]]).to(self.fc1.weight.dtype)
    def set_radius_nnrepair(self):
        self.radius = 0.5
    def set_radius_deeppoly(self):
        self.radius = 0.5
if __name__ == '__main__':
    device = torch.device(f'cuda:3')
    net = ExampleNet(options='deeppoly').to(device)
    net.eval()
    example_input = net.inputs().to(device)
    print(net(example_input))

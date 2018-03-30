# -*- coding: utf-8 -*-
__author__ = 'Zhouhao Zeng'

import torch
import torch.nn as nn
import torch.nn.functional as F

# Network for simple task (i.e. CartPole) with value based methods (i.e. DQN)
class FCNet(nn.Module):
    def __init__(self, dims):
        super(FCNet, self).__init__()
        self.fc_list = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.fc_list.append(nn.Linear(in_dim, out_dim))

        self.parameters = nn.ParameterList(reduce(lambda x, y: x + list(y.parameters()), self.fc_list, []))

    def forward(self, x):
        for fc in self.fc_list[:-1]:
            x = F.relu(fc(x))
        return self.fc_list[-1](x)


# Network for simple task (i.e. CartPole) with REINFORCE
class REINFORCEFCNet(FCNet):
    def __init__(self, dims):
        FCNet.__init__(self, dims)

    def forward(self, x):
        for fc in self.fc_list[:-1]:
            x = F.relu(fc(x))
        return F.softmax(self.fc_list[-1](x), dim=1)
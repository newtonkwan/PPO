import torch 
from torch import nn
import torch.nn.functional as F 
import numpy as np 

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim): 
        super(FeedForwardNN, self).__init__() # inherit from nn.Module class

        # define some nn layers 
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

        forward.







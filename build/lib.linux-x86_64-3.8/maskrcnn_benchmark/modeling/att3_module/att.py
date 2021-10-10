# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
import pdb
class ATTModule(torch.nn.Module):
    """
    Module for ATT computation. Takes feature maps from the backbone and outputs 
    ATT proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(ATTModule, self).__init__()

        self.cfg = cfg.clone()
        self.data_num = cfg.MODEL.ROI_BOX_HEAD.NUM_DATA
        self.fc1 = []
        self.fc2 = []
        self.relu = nn.ReLU()
        self.channels = in_channels
        self.adt_num = cfg.MODEL.ADT_NUM
        self.attention = nn.Parameter(torch.ones(1,in_channels), requires_grad=True)        
    def forward(self, features):
        
        new_score = torch.unsqueeze(self.attention,2)
        new_score = torch.unsqueeze(new_score,2)
        new_score = new_score.repeat(features.size(0),1,features.size(2),features.size(3))
        new_features = features*new_score
        return new_features
      
                 

def build_att(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return ATTModule(cfg, in_channels)

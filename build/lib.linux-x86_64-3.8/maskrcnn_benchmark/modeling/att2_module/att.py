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
        
        self.fc = nn.Linear(in_channels,self.adt_num)
        for i in range(self.adt_num):

              self.fc1.append(nn.Linear(in_channels,int(in_channels/4)))
              self.fc2.append(nn.Linear(int(in_channels/4), in_channels))
        self.fc1 = nn.ModuleList(self.fc1)
        self.fc2 = nn.ModuleList(self.fc2)
        self.soft = nn.Softmax(1)
        self.sig = nn.Sigmoid()
    def forward(self, features):
        features1 = features.view(features.size(0),features.size(1),-1)
        pooled_features = torch.mean(features1,2)
        self.feat_list = []
        feat = self.fc1[0](pooled_features)
        feat = self.relu(feat)
        feat = self.fc2[0](feat)
        new_score = torch.unsqueeze(feat,2)
        new_score = torch.unsqueeze(new_score,2)
        new_score = new_score.repeat(1,1,features.size(2),features.size(3))
        new_features = features*new_score
        return new_features
      
                 

def build_att(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return ATTModule(cfg, in_channels)

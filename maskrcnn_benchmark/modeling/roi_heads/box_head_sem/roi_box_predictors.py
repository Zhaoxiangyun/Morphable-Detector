# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import torch.nn.functional as F
from torch import nn
import pdb
import torch
import numpy as np
@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, 1)
        
       
        self.cls_logits = nn.Conv2d(
             self.feature_size, num_classes, kernel_size=1, stride=1
        )

        torch.nn.init.constant_(self.cls_logits.bias, 0)
        params = self.cls_logits.parameters()
        for param in params:
              param.requires_grad = False
        self.cls_sem = nn.Linear(2048, self.feature_size)
 
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        score_b = self.cls_score(x)
        sem = self.cls_sem(x)
        score = self.cls_logits(sem)

        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
       
        self.num_classes = num_classes
        representation_size = in_channels
        self.cls_score = nn.Linear(representation_size, 1)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.feature_size = cfg.FEATURE_SIZE
        
        self.scale = cfg.TEST_SCALE
        sem_matrix_v = np.ones((num_classes-1,self.feature_size))
        sem_matrix_l = np.ones((num_classes-1,self.feature_size))

        self.get_feature = cfg.GET_FEATURE
        if cfg.VISUAL: 
           f = open(cfg.SEM_DIR,'r')
           lines = f.readlines()
           for i in range(len(lines)):
              words = lines[i].split(',')
              vec = np.array(words)
              sem_matrix_v[i,:] = vec
           f.close()    
           sem_matrix = sem_matrix_v
        else:   
          f = open(cfg.LOAD_SEM_DIR,'r')
          lines = f.readlines()
          f.close() 
          for i in range(num_classes-1):
            words = lines[i].split(' ')
            words.remove(words[0])
            vec = np.array(words)
            sem_matrix_l[i,:] = vec[0:self.feature_size]       
          sem_matrix = sem_matrix_l 

        self.cls_sem = nn.Linear(representation_size, self.feature_size)                  
        sem_matrix = torch.from_numpy(sem_matrix)
        sem_matrix = sem_matrix.type(torch.cuda.FloatTensor)
        self.sem_matrix = sem_matrix
        

        nn.init.normal_(self.cls_sem.weight,std=0.01)
        nn.init.constant_(self.cls_sem.bias, 0)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)
    def forward(self, x):
        
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        b_score = self.cls_score(x)
        sem = self.cls_sem(x)
        if self.get_feature:
            return sem
        if self.training:
           self.sem_matrix = F.normalize(self.sem_matrix, p=2, dim=1)
        else:
           self.sem_matrix  = self.scale*F.normalize(self.sem_matrix, p=2, dim=1)        
        cls_score = []
        for i in range(1,self.num_classes):
            sem_i = self.sem_matrix[i-1,:]
            sem_i = sem_i.repeat(sem.size(0),1)
            sem_i = sem_i.type(torch.cuda.FloatTensor)
            score = torch.sum(torch.mul(sem,sem_i),1)
            score = score.unsqueeze(1)
            cls_score.append(score)
        

        cls_score = torch.cat(cls_score,1)
        scores = torch.cat((b_score,cls_score),1)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)

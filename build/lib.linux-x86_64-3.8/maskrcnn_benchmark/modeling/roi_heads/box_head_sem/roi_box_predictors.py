# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
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
        
        self.size = 400
        self.anchor_num = num_anchors
        f = open('semantic_vector_coco_400.txt','r')
        lines = f.readlines()
        sem_matrix = np.ones((80,self.size))
        for i in range(len(lines)):
            words = lines[i].split(' ')
            words.remove(words[0])
            vec = np.array(words)
            sem_matrix[i,:] = vec
        
        self.cls_logits = nn.Conv2d(
             self.size, num_classes, kernel_size=1, stride=1
        )

        sem_matrix = torch.from_numpy(sem_matrix)
        
        sem_matrix = sem_matrix.unsqueeze(2)
        sem_matrix = sem_matrix.unsqueeze(2)

        self.cls_logits.weight.data = sem_matrix
        torch.nn.init.constant_(self.cls_logits.bias, 0)
        params = self.cls_logits.parameters()
        for param in params:
              param.requires_grad = False
        self.cls_sem = nn.Linear(2048, self.size)
 
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
    def __init__(self, cfg, in_channels, ignore_labels=None):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        if num_classes > 30:
            self.coco = True
        else:
            self.coco = False
        self.num_classes = num_classes
        representation_size = in_channels
        self.cls_score = nn.Linear(representation_size, 1)
        self.ignore_labels = cfg.IGNORE_LABEL
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.size = 200
        
        sem_matrix_v = np.ones((num_classes-1,self.size))
        sem_matrix_l = np.ones((num_classes-1,200))

        self.att = []
        self.multi = cfg.MULTI
        self.get_feature = cfg.FEATURE
        
        f = open(cfg.SEM_DIR,'r')
        lines = f.readlines()
        for i in range(len(lines)):
            words = lines[i].split(',')
            vec = np.array(words)
            sem_matrix_v[i,:] = vec
        f.close()     
        if num_classes>100:
            f = open('800_sem.txt','r')
        else:
            f = open('semantic_vector_coco.txt','r')
        lines = f.readlines()
        f.close()
        for i in range(len(lines)):
          words = lines[i].split(' ')
          words.remove(words[0])
          vec = np.array(words)
          sem_matrix_l[i,:] = vec[0:200]
        if cfg.VISUAL:
          sem_matrix = sem_matrix_v
        else:
          sem_matrix = sem_matrix_l  
        if cfg.MULTI:
            self.emb = []
            self.mapping = {}
            group_list = []
            group = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 78]
            group_list.append(group)
            group = [43, 44, 45, 65, 66, 67, 68, 74, 77, 79, 80]
            group_list.append(group)
            group = [30, 33, 40, 41, 42, 46, 59, 75, 76]
            group_list.append(group)
            group = [14, 29, 57, 58, 60, 61, 62, 63, 64, 69, 70, 71, 72, 73]
            group_list.append(group)
            group = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 34, 38]
            group_list.append(group)
            group = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
            group_list.append(group)
            group = [25, 27, 28, 31, 32, 35, 36, 37, 39]
            group_list.append(group)
            assert(len(group_list)==7)
          #  group = list(range(30,40))
          #  group_list.append(group)
          #  
          #  group = list(range(25,30))
          #  group_list.append(group)
          #  
          #  group= list(range(15,25))
          #  group_list.append(group)
          #  
          #  group= list(range(10,15))
          #  group_list.append(group)
          #  
          #  group= list(range(2,10))
          #  group_list.append(group)

          #  group= [1]
          #  group_list.append(group)

          #  group= list(range(74,81))
          #  group_list.append(group)

          #  group= list(range(69,74))
          #  group_list.append(group)

          #  group= list(range(63,69))
          #  group_list.append(group)

          #  group= list(range(57,63))
          #  group_list.append(group)

          #  group= list(range(47,57))
          #  group_list.append(group)

          #  group= list(range(40,47))
          #  group_list.append(group)
          #  assert(len(group_list)==12) 
            self.mapping = {}
            for gr in range(len(group_list)):
                for g in group_list[gr]:
                    self.mapping[g] = gr
            for i in range(len(group_list)):
              self.emb.append(nn.Linear(representation_size, self.size))

        else:
                self.cls_sem = nn.Linear(representation_size, self.size)
        if cfg.FINE:
               self.fc = []
               for i in range(5):
                   self.fc.append(nn.Linear(200, 1))
               self.fc = nn.ModuleList(self.fc)    
               req = True
               self.att = nn.Parameter(torch.from_numpy(sem_matrix),requires_grad=req)

       # if cfg.FINE:
       #    self.att = nn.ParameterList(self.att)
        if cfg.MULTI:
          self.emb = nn.ModuleList(self.emb)

        self.finetune = cfg.FINE   
       #self.cls_logits = nn.Conv2d(
       #      self.size, num_classes-1, kernel_size=1, stride=1
       # )
           
        sem_matrix = torch.from_numpy(sem_matrix)
#        sem_matrix = sem_matrix.unsqueeze(2)
#        sem_matrix = sem_matrix.unsqueeze(2)
        sem_matrix = sem_matrix.type(torch.cuda.FloatTensor)
        self.sem_matrix = sem_matrix
#        self.cls_logits.weight.data = sem_matrix
#        torch.nn.init.constant_(self.cls_logits.bias, 0)
#        params = self.cls_logits.parameters()
#        for param in params:
#              param.requires_grad = False
        if self.multi:
           for mod in self.emb:
               nn.init.normal_(mod.weight,std=0.01)
               nn.init.constant_(mod.bias, 0)
        else:
            nn.init.normal_(self.cls_sem.weight,std=0.01)
            nn.init.constant_(self.cls_sem.bias, 0)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        if self.finetune:
          for l in self.fc:
              nn.init.constant_(l.bias, 0)
              nn.init.normal_(l.weight, std=0.01)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)
        if False:
          for l in [self.bbox_pred]:
              for param in l.parameters():
                  param.requires_grad = False
    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        if self.finetune:    
            x = x.detach()
        b_score = self.cls_score(x)
        if not self.multi:
           sem = self.cls_sem(x)
        cls_score = []
        if True:
           ignore_labels = self.ignore_labels
           #ignore_labels = [2,3,7,17,57]
           ignore_labels = list(range(0,80))  
           ignore_labels2 = [2,3,17,19,48,38,69,54,64,73,30]
        else:    
           ignore_labels = self.ignore_labels +[8, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
           ignore_labels2 = [8, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
        for i in range(1,self.num_classes):
          if self.multi:
            sem = self.emb[self.mapping[i]](x)      
          if i in ignore_labels:
            if self.finetune:
                if i in self.ignore_labels:   
                 #score = self.fc[self.ignore_labels.index(i)](sem)
                  sem_i = self.att[i-1,:]
                  sem_i = sem_i.repeat(sem.size(0),1)
                  sem_i = sem_i.type(torch.cuda.FloatTensor)
                  score = torch.mean(torch.mul(sem.detach(),sem_i),1)
               
                else: 
                  score = torch.zeros(score.size())
                  score = score.type(torch.cuda.FloatTensor)
 
                # sem_i = self.sem_matrix[i-1,:]
            
            else:
              sem_i = self.sem_matrix[i-1,:]
              sem_i = sem_i.repeat(sem.size(0),1)
              sem_i = sem_i.type(torch.cuda.FloatTensor)
              score = torch.mean(torch.mul(sem.detach(),sem_i),1)
            if self.training:
                if self.finetune:
                    score = torch.mean(torch.mul(sem,sem_i),1)
                else:
                  score = torch.zeros(score.size())
                  score = score.type(torch.cuda.FloatTensor)
            if i not in ignore_labels2:
                score = torch.zeros(score.shape[0])
                score = score.type(torch.cuda.FloatTensor)

#            if self.finetune:
#                sem_i = self.att[i-1,:]
#                sem_i.repeat(sem.size(0),1)
#                score = torch.mean(torch.mul(sem.detach(),sem_i),1)
          else:
            sem_i = self.sem_matrix[i-1,:]
            sem_i = sem_i.repeat(sem.size(0),1)
            sem_i = sem_i.type(torch.cuda.FloatTensor)
            score = torch.mean(torch.mul(sem,sem_i),1)
          score = score.unsqueeze(1)
          cls_score.append(score)
        #sem = sem.unsqueeze(2)
        #sem = sem.unsqueeze(2)
        cls_score = torch.cat(cls_score,1)
        scores = torch.cat((b_score,cls_score),1)
        bbox_deltas = self.bbox_pred(x)
        if self.get_feature:
            return sem
        return scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels, ignore_labels=None):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels, ignore_labels=ignore_labels)

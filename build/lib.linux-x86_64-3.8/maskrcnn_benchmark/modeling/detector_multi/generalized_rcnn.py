# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn_domain.rpn import build_rpn
from ..roi_heads.roi_heads_adaptive import build_roi_heads
from ..att3_module.att import build_att
import pdb
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.num_data = cfg.MODEL.ROI_BOX_HEAD.NUM_DATA
        self.uod = cfg.MODEL.UOD
        self.rpn_list = []
        self.head_list = []
        self.domain = cfg.MODEL.DOMAIN
        num_list = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASS_LIST
        self.att0 = build_att(cfg, 256) 
        self.att1 = build_att(cfg, 256) 
        self.att2 = build_att(cfg, 256) 
        self.att3 = build_att(cfg, 256) 
        self.att4 = build_att(cfg, 256) 

        self.att5 = build_att(cfg, 256) 
        self.att6 = build_att(cfg, 256) 
        self.att7 = build_att(cfg, 256) 
        self.att8 = build_att(cfg, 256) 
        self.att9 = build_att(cfg, 256) 



        for i in range(self.num_data):
            self.rpn_list.append(build_rpn(cfg, self.backbone.out_channels))
            self.head_list.append(build_roi_heads(cfg, self.backbone.out_channels, class_num=num_list[i]))
        self.rpn_list =  nn.ModuleList(self.rpn_list)   
        self.head_list = nn.ModuleList(self.head_list)
#        self.rpn = build_rpn(cfg, self.backbone.out_channels)
#        self.rpn1 = build_rpn(cfg, self.backbone.out_channels)
#        self.rpn2 = build_rpn(cfg, self.backbone.out_channels)
#        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, data_index=0, ignore_labels=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dic:t[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        features_list = []

        if True: 
            features = list(features)
            features1 = []
            features1.append(self.att0(features[0]))
            features1.append(self.att1(features[1]))
            features1.append(self.att2(features[2]))
            features1.append(self.att3(features[3]))
            features1.append(self.att4(features[4]))
            features1 = tuple(features1)
            features_list.append(features1)

            features2 = []
            features2.append(self.att0(features[0]))
            features2.append(self.att1(features[1]))
            features2.append(self.att2(features[2]))
            features2.append(self.att3(features[3]))
            features2.append(self.att4(features[4]))
            features2 = tuple(features2)
            features_list.append(features2)


        proposals_list = []
        proposal_losses_list = []
        domain_list = []
        if self.training:
          for i in range(self.num_data):
              proposals, proposal_losses, domain = self.rpn_list[i](images, features_list[i], targets)
              proposals_list.append(proposals)
              proposal_losses_list.append(proposal_losses)
         #     domain_list.append(domain)
          proposals = proposals_list[data_index]
          proposal_losses = proposal_losses_list[data_index]
          
         # domain = torch.cat(domain_list,1)
         # domain_labels=data_index*torch.ones(domain.shape[0])
         # domain_labels = torch.squeeze(domain_labels)
         # domain_labels = domain_labels.type(torch.LongTensor).cuda()
         # if self.domain:
         #     classification_loss = F.cross_entropy(domain, domain_labels)
#        #      F.binary_cross_entropy_with_logits()+ F.binary_cross_entropy_with_logits
         # else: 
         #     classification_loss = 0*F.cross_entropy(domain, domain_labels)

 # domain classification loss
#           domain_list = []
#           
#           for i in range(self.num_data):
#              domain = self.rpn_list[i](images, features, targets, get_domain=True)
#              domain_list.append(domain)
#           for i in range(self.num_data):
#               domain_score_list = []
#               for j in range(len(domain)):
#                 domain_score = domain_list[i][j]
#                 domain_score = domain_score.repeat(1,3,1,1)
#                 domain_score_list.append(domain_score)
#               proposals, proposal_losses, domain = self.rpn_list[i](images, features, targets,domain_score=domain_score_list)
#               proposal_losses_list.append(proposal_losses)
#               proposals_list.append(proposals)
#           proposals = proposals_list[data_index]
#           proposal_losses = proposal_losses_list[data_index]
#           score_list = []
#           classification_loss = 0
#           for p in range(len(proposals_list)):
#               pro = proposals_list[p]
#               sc = []
#               for p1 in range(len(pro)):
#                   sc.append(pro[p1].get_field('domain_score'))
#               score = torch.cat(sc)
#               if p == data_index:
#                   objective = torch.ones(score.shape)
#               else:
#                   objective = torch.zeros(score.shape)
#               objective = objective.cuda()    
#               classification_loss = classification_loss + F.binary_cross_entropy_with_logits(score, objective)
#           
        else:   
           proposals, proposal_losses, domain = self.rpn_list[data_index](images, features_list[data_index], targets)

        th = 0.0
        if self.head_list:
            if not ignore_labels:
                if not self.training:
       
                  
                   proposals, proposal_losses, domain = self.rpn_list[data_index](images, features_list[data_index], targets)

                   x, result0, detector_losses = self.head_list[data_index](features_list[data_index], proposals, targets)
                   result = result0

       
 
#                   proposals, proposal_losses, domain = self.rpn_list[1](images, features, targets,domain_score=domain_score_list)
#
#                   x, result, detector_losses = self.head_list[1](features, proposals, targets)
#                   for m in range(len(result)):
#                     result[m] = cat_boxlist((result[m],result0[m]))
                else:
                   x, result, detector_losses = self.head_list[data_index](features_list[data_index], proposals, targets)
            else:
              x, result, detector_losses = self.head_list[data_index](features_list[data_index], proposals, targets=targets, ignore_labels=ignore_labels)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

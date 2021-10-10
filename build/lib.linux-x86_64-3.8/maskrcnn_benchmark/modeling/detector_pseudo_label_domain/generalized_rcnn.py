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
        self.rpn_list = []
        self.head_list = []
        self.domain = cfg.MODEL.DOMAIN
        num_list = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASS_LIST
        for i in range(self.num_data+1):
            self.rpn_list.append(build_rpn(cfg, self.backbone.out_channels))
            self.head_list.append(build_roi_heads(cfg, self.backbone.out_channels, class_num=num_list[i]))
        self.rpn_list =  nn.ModuleList(self.rpn_list)   
        self.head_list = nn.ModuleList(self.head_list)
#        self.rpn = build_rpn(cfg, self.backbone.out_channels)
#        self.rpn1 = build_rpn(cfg, self.backbone.out_channels)
#        self.rpn2 = build_rpn(cfg, self.backbone.out_channels)
#        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, data_index=0, ignore_labels=None, iteration=None):
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
        proposals_list = []
        proposal_losses_list = []
        domain_list = []
        if self.training:
          for i in range(self.num_data+1):
              proposals, proposal_losses, domain = self.rpn_list[i](images, features, targets)
              proposals_list.append(proposals)
              proposal_losses_list.append(proposal_losses)
              domain_list.append(domain)
          proposal_losses_1 = {}
          proposal_losses_1['loss_objectness_1'] = proposal_losses_list[0]['loss_objectness']
          proposal_losses_1['loss_rpn_box_reg_1'] = proposal_losses_list[0]['loss_rpn_box_reg']

#          domain = torch.cat(domain_list,1)
#          domain_labels=data_index*torch.ones(domain.shape[0])
#          domain_labels = torch.squeeze(domain_labels)
#          domain_labels = domain_labels.type(torch.LongTensor).cuda()
#          # domain classification loss
#          if self.domain:
#            classification_loss = F.cross_entropy(domain, domain_labels)
#          else: 
#              classification_loss = 0*F.cross_entropy(domain, domain_labels)
#        else:   
#           domain_list = []
#           
#           for i in range(self.num_data):
#              domain = self.rpn_list[i](images, features, targets, get_domain=True)
#              domain_list.append(domain)
#           domain_score_list = []
           for branch_index in range(2):
            for j in range(len(domain)):
              values, index = torch.max(torch.cat((domain_list[0][j],domain_list[1][j]),1),1)
              domain_score = index == brach_index
              domain_score = torch.unsqueeze(domain_score,1)
              domain_score = domain_score.repeat(1,3,1,1)
              domain_score_list.append(domain_score)
            proposals, proposal_losses, domain = self.rpn_list[branch_index](images, features, targets,domain_score=domain_score_list)
            x, result, detector_losses = self.head_list[0](features, proposals, targets=targets, ignore_labels=ignore_labels)   


        if self.head_list:
            if not ignore_labels:
                if not self.training:
#                   domain_score_list = []
#                   for j in range(len(domain)):
#                     values, index = torch.max(torch.cat((domain_list[0][j],domain_list[1][j]),1),1)
#                     domain_score = index == 0
#                     domain_score = torch.unsqueeze(domain_score,1)
#                     domain_score = domain_score.repeat(1,3,1,1)
#                     domain_score_list.append(domain_score)
#                   
                   proposals, proposal_losses, domain = self.rpn_list[1](images, features, targets)

                   x, result0, detector_losses = self.head_list[1](features, proposals, targets)
#                   domain_score_list = []
#                   for j in range(len(domain)):
#                     values, index = torch.max(torch.cat((domain_list[0][j],domain_list[1][j]),1),1)
#                     domain_score = index == 1
#                     domain_score = torch.unsqueeze(domain_score,1)
#                     domain_score = domain_score.repeat(1,3,1,1)
#                     domain_score_list.append(domain_score)
 
                   proposals, proposal_losses, domain = self.rpn_list[2](images, features, targets)

                   x, result, detector_losses = self.head_list[2](features, proposals, targets)
                   for m in range(len(result)):
                     result[m] = cat_boxlist((result[m],result0[m]))
                else:
                   x, result, detector_losses = self.head_list[data_index](features, proposals_list[data_index], targets=targets)
                   detector_losses1 = {}
                   detector_losses1['loss_classifier_1'] = detector_losses['loss_classifier']
                   detector_losses1['loss_box_reg_1'] = detector_losses['loss_box_reg']
                   detector_losses2 = {}
                   if iteration is not None:
                    if iteration > 0:
                     if data_index == 100:
                       for k in range(len(self.head_list)):
                          if k == 0:
                              continue
                          if k == data_index:
                              continue
                         
                          x, result, detector_losses = self.head_list[k](features, proposals_list[k], targets=targets, get_det=True)
                          for k1 in range(len(targets)):
                              if bool(result[k1]):
                                  
                                result[k1].add_field('labels',result[k1].get_field('labels').type(torch.float))
                                targets[k1].add_field('scores', result[k1].get_field('scores'))
                                result[k1].add_field('difficult', targets[k1].get_field('difficult'))
                                targets[k1] = cat_boxlist((targets[k1],result[k1]))
                              else:
                                continue
                     x, result, detector_losses2 = self.head_list[0](features, proposals_list[0], targets=targets)
#
            else:
              x, result, detector_losses = self.head_list[0](features, proposals, targets=targets, ignore_labels=ignore_labels)
  
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses1)
            losses.update(detector_losses2)
            losses.update(proposal_losses_list[data_index])
            losses.update(proposal_losses_1)

#    losses.update(dict(loss_domain=classification_loss))

            return losses

        return result

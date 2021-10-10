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
from ..att_module.att import build_att

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
        self.unify = cfg.MODEL.UNIFY
        self.seperate = cfg.MODEL.SEP
        self.pseudo = cfg.MODEL.PSEUDO
        self.uod = cfg.MODEL.UOD
        self.att = build_att(cfg, 256)
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
        if self.uod: 
            features = list(features)
            features[0] = self.att(features[0])
            features[1] = self.att(features[1])
            features[2] = self.att(features[2])
            features[3] = self.att(features[3])
            features[4] = self.att(features[4])
            features = tuple(features)
        proposals_list = []
        proposal_losses_list = []
        domain_list = []
        if self.training:
          for i in range(1):
              proposals, proposal_losses, domain = self.rpn_list[i](images, features, targets)
              proposals_list.append(proposals)
              proposal_losses_list.append(proposal_losses)
          # pseudo detection
          if self.pseudo:
             data_index_t = 1
          else:
             data_index_t = 1000                            

  
          proposal_losses_1 = {}

          if not self.seperate:
             proposal_losses_1 = {}
             proposal_losses_1['loss_objectness_1'] = proposal_losses_list[0]['loss_objectness']
             proposal_losses_1['loss_rpn_box_reg_1'] = proposal_losses_list[0]['loss_rpn_box_reg']
          else:
             proposal_losses_1['loss_objectness_1'] = 0*proposal_losses_list[0]['loss_objectness']
             proposal_losses_1['loss_rpn_box_reg_1'] = 0*proposal_losses_list[0]['loss_rpn_box_reg']
 
        if self.head_list:
            if not ignore_labels:
                if not self.training:
                     if not self.unify:               
                          proposals, proposal_losses, domain = self.rpn_list[1](images, features, targets)
                          det_label = [1,2,3,4,5,6,7,9,12,15,16,17,18,19,20,40,57,58,59,61,63]
                          x, result, detector_losses = self.head_list[1](features, proposals, targets=targets, get_det=True, det_labels=det_label)
                      
                     else:
                          proposals, proposal_losses, domain = self.rpn_list[0](images, features, targets)
       
                          x, result, detector_losses = self.head_list[0](features, proposals, targets)
 
                else:
                     x, result, detector_losses2 = self.head_list[0](features, proposals_list[0], targets=targets)
# igniore labels
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses2)
            losses.update(proposal_losses_list[0])

#    losses.update(dict(loss_domain=classification_loss))

            return losses

        return result

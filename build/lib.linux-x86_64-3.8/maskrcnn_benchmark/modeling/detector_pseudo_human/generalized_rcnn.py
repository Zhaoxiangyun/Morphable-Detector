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
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms



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
        self.finetune = cfg.FINE

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
        if self.finetune:
          for param in self.backbone.parameters():
            param.requires_grad = False
          for i in range(1,self.num_data+1):
            for param in self.rpn_list[i].parameters():
              param.requires_grad = False
            for param in self.head_list[i].parameters():
              param.requires_grad = False



    def forward(self, images, targets=None, targets2=None, data_index=0, ignore_labels=None, iteration=None):
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
          for i in range(self.num_data+1):
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
       
                          det = False
                          x, result0, detector_losses = self.head_list[1](features, proposals, targets,get_det=False, det_labels=[1])
                          result = result0
                          proposals, proposal_losses, domain = self.rpn_list[2](images, features, targets)
       
                          x, result, detector_losses = self.head_list[2](features, proposals, targets, get_det=False, det_labels=[84])
                          for m in range(len(result)):
                            result[m] = cat_boxlist((result[m],result0[m]))
                          proposals, proposal_losses, domain = self.rpn_list[3](images, features, targets)
       
                          x, result3, detector_losses = self.head_list[3](features, proposals, targets)
                          for m in range(len(result)):
                            result[m] = cat_boxlist((result[m],result3[m]))
#                             
#                            new_result = []
#                            for i in range(1,86):
#                              labels = result[m].get_field('labels')
#                              ind = labels == i
#                              new_result.append(boxlist_nms(result[m][ind],0.5))
#                            new_result = cat_boxlist(new_result)
#
#                            result[m] = new_result 


                     else:
                          proposals, proposal_losses, domain = self.rpn_list[0](images, features, targets)
       
                          x, result, detector_losses = self.head_list[0](features, proposals, targets)
                else:
                   if self.seperate:   
                       x, result, detector_losses = self.head_list[data_index](features, proposals_list[data_index], targets=targets)
                       detector_losses1 = {}
                       detector_losses1['loss_classifier_1'] = detector_losses['loss_classifier']
                       detector_losses1['loss_box_reg_1'] = detector_losses['loss_box_reg']
                       detector_losses2 = {}
                   result_all = None
                   if not self.seperate:
                     if False:       
                        for k in range(len(self.head_list)):
                          if k == 0:
                              continue
                          if k==1:
                             det_label = [1]
                          if k==2:
                              det_label = [84]
                          if k==data_index:
                              continue
                          x, result, detector_losses = self.head_list[k](features, proposals_list[k], targets=targets, get_det=True, det_labels=det_label)
                          if result_all is None:
                           for k1 in range(len(result)):
                             index = False  
                             if bool(result[k1]):
                                result[k1].add_field('labels',result[k1].get_field('labels').type(torch.float))
                                index= True
                           if index:      
                             result_all = result
                          else:     
                           for k1 in range(len(result)):
                              if bool(result[k1]):
                                result[k1].add_field('labels',result[k1].get_field('labels').type(torch.float))
                                if bool(result_all[k1]): 
                                   result_all[k1] = cat_boxlist((result_all[k1],result[k1]))
                                else:   
                                   result_all[k1] = result[k1]

                              else:
                                continue
                       # if result_all is not None:
                        NMS = False
                        if NMS:
                         if result_all is not None:
                          for m in range(len(result_all)):
                             
                            new_result = []
                            
                            if bool(result_all[m]):  
                              for i in range(1,2):
                                labels = result_all[m].get_field('labels')
                                ind = labels == i
                                new_result.append(boxlist_nms(result_all[m][ind],0.5))
                            else:  
                                  continue
                            new_result = cat_boxlist(new_result)

                            result_all[m] = new_result 

                           
                     x, result, detector_losses2 = self.head_list[0](features, proposals_list[0], targets=targets, targets2=targets2)
# igniore labels
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:

            losses = {}

            if not self.seperate:
               losses.update(detector_losses2)
               losses.update(proposal_losses_1)

            else:    
        
              losses.update(detector_losses1)
              losses.update(proposal_losses_list[data_index])


#    losses.update(dict(loss_domain=classification_loss))

            return losses

        return result

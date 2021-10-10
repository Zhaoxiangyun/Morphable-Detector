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
        self.class_num = num_list[0]
        self.unify = cfg.MODEL.UNIFY
        self.seperate = cfg.MODEL.SEP
        self.pseudo = cfg.MODEL.PSEUDO
        self.uod = cfg.MODEL.UOD
        if self.pseudo:
            self.rpn_list.append(build_rpn(cfg, self.backbone.out_channels))
            self.head_list.append(build_roi_heads(cfg, self.backbone.out_channels, class_num=num_list[0]))
        else:
          for i in range(self.num_data+1):
            self.rpn_list.append(build_rpn(cfg, self.backbone.out_channels))
            self.head_list.append(build_roi_heads(cfg, self.backbone.out_channels, class_num=num_list[i]))

        self.rpn_list =  nn.ModuleList(self.rpn_list)   
        self.head_list = nn.ModuleList(self.head_list)
        self.att = build_att(cfg, 256)

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
            features1 = list(features)
            features1[0] = self.att(features1[0])
            features1[1] = self.att(features1[1])
            features1[2] = self.att(features1[2])
            features1[3] = self.att(features1[3])
            features1[4] = self.att(features1[4])
            features1 = tuple(features1)
        proposals_list = []
        proposal_losses_list = []
        domain_lisdt = []

        if self.training:
          if not self.seperate:
              proposals, proposal_losses, domain = self.rpn_list[0](images, features, targets)
  
              proposals_list.append(proposals)
              proposal_losses_list.append(proposal_losses)

         
          else:    
            proposals_list.append([])
            proposal_losses_list.append([])


            for i in range(1,self.num_data+1):
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
             proposal_losses_1['loss_objectness_1'] = 0
             proposal_losses_1['loss_rpn_box_reg_1'] = 0
     
        if self.head_list:
            if not ignore_labels:
                if not self.training:
                     if not self.unify: 
                         #data index 1
                          det_label1 = [-1,14,46,96,75,42,65,67,70,72,88,10,56,92,25,27,43,44,74,87,69,45,76]
                         
                          det = False
                         #data index 2
                         # det_label1 = [-1,1,2,3,4,5,6,7,9,15,16,17,18,19,20,40,57,58,59,61,63,92,88,96]
                         # data index 3 
                        #  det_label1 = [-1,3,8,10,12,92,5,1,90,7,8,11,14]
                         # data index 4 
                         # det_label1 = [-1,1,3,10,25,90,6,8,11,81,92,82,12,2,57,4,27,14,7] 

                         # data index 5 
                         # det_label1 = [-1,1,44,43,74,42,66,41,59,45,14,25,64,40,76,27,67,77,46,72,65,69,29,2,73]
          
                         # head 1
                         # det_label2 = [1,2,3,4,5,6,7,9,15,16,17,18,19,20,40,57,58,59,61,63]
                         # det_label =  [x for x in det_label1 if x in det_label2]

                         #   
                         # proposals, proposal_losses, domain = self.rpn_list[1](images, features, targets)
                         # x, result0, detector_losses = self.head_list[1](features, proposals, targets, get_det=det, det_labels=det_label)
                         # result = result0
                          # head 2
                          det_label2 = [8, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
                          det_label =  [x for x in det_label1 if x in det_label2]

                          proposals, proposal_losses, domain = self.rpn_list[1](images, features, targets)
       
                          x, result, detector_losses = self.head_list[1](features, proposals, targets)
                           
                        #  for m in range(len(result)):
                        #    if not bool(result[m]):
                        #         result[m] = result0[m]
                        #         continue
                        #    if not bool(result0[m]):
                        #          continue
                        #    result[m] = cat_boxlist((result[m],result0[m]))
                        #  # head 3   
                        #  det_label2 = [12,81,82,83]

                        #  det_label = [x for x in det_label1 if x in det_label2]


                        #  proposals, proposal_losses, domain = self.rpn_list[3](images, features, targets)
        
                        #  x, result3, detector_losses = self.head_list[3](features, proposals, targets, get_det=det, det_labels=det_label)
                        #  for m in range(len(result)):
                        #    
                        #    if not bool(result[m]):
                        #         result[m] = result3[m]
                        #         continue
                        #    if not bool(result3[m]):
                        #          continue
                        #    result[m] = cat_boxlist((result[m],result3[m]))
                          # head 4
                        #  proposals, proposal_losses, domain = self.rpn_list[4](images, features, targets)
       
                        #  x, result4, detector_losses = self.head_list[4](features, proposals, targets, get_det=True, det_labels=det_label)
                        #  for m in range(len(result)):
                        #    result[m] = cat_boxlist((result[m],result4[m]))
                          # head 5
                       #   det_label2 = [57,58,60,62,63,72]+list(range(85,98))
                       #   det_label =  [x for x in det_label1 if x in det_label2]


                       #   proposals, proposal_losses, domain = self.rpn_list[5](images, features, targets)
       
                       #   x, result5, detector_losses = self.head_list[5](features, proposals, targets, get_det=det, det_labels=det_label)
                       #   for m in range(len(result)):
                       #     if not bool(result[m]):
                       #          result[m] = result5[m]
                       #          continue
                       #     if not bool(result5[m]):
                       #           continue
                       #     result[m] = cat_boxlist((result[m],result5[m]))
                       #   for m in range(len(result)):
                       #    
                       #     new_result = []
                       #     if det:
                       #         det_label_final = det_label1
                       #     else:
                       #         det_label_final = list(range(1,98))
                       #     for i in det_label_final:
                       #       labels = result[m].get_field('labels')
                       #       ind = labels == i
                       #       if i == -1:
                       #          new_result.append(result[m][ind])
                       #       else: 
                       #          new_result.append(boxlist_nms(result[m][ind],0.5))
                       #       #new_result.append(result[m][ind])

                       #     new_result = cat_boxlist(new_result)

                       #     result[m] = new_result 





                           #   result[m] = boxlist_nms(result[m],0.5) 


                     else:
                       if self.uod:  
                          used_feature = features1
                       else:
                          used_feature = features

                       proposals, proposal_losses, domain = self.rpn_list[0](images, used_feature, targets)
       
                       x, result, detector_losses = self.head_list[0](used_feature, proposals, targets)
                      
                else:
                   if self.seperate:
                      x, result, detector_losses = self.head_list[data_index](features, proposals_list[data_index], targets=targets)
                      detector_losses1 = {}
                      detector_losses1['loss_classifier_1'] = detector_losses['loss_classifier']
                      detector_losses1['loss_box_reg_1'] = detector_losses['loss_box_reg']
                   if not self.seperate:

                     result_all = None
                     if False:       
                        if data_index == 3:
                           det_label1 = [3,8,12,92,5,1,90,7,8,11,14]
                        if data_index == 1:
                           det_label1 = [14,46,96,75,42,65,67,70,72,88,10,56,92,25,27,43,44,74,87,69,45,76]
                        if data_index == 5:
                            det_label1 = [44,43,74,42,66,41,59,45,14,25,64,40,76,27,67,77,46,72,65,69,29,2,73]
                            #det_label1 = []
                        if data_index == 4:
                           # det_label1 = [10,25,90,6,8,11,81,92,82,12,2,57,4,27,14,7] 
                            pass
                        if data_index == 2:
                            det_label1 = [1,2,3,4,5,6,7,9,15,16,17,18,19,20,40,57,58,59,61,63,92,88,96]
                        for k in range(len(self.head_list)):
                          if k == 0:
                              continue
                          if k == 3:
                              det_label2 = [12,81,82,83]
                          if k == 1:
                              det_label2 = [1,2,3,4,5,6,7,9,15,16,17,18,19,20,40,57,58,59,61,63]
                          if k == 5:
                              det_label2 = [57,58,60,62,63,72]+list(range(85,98))
                             #det_label2 = list(range(85,98))

                          if k == 4:
                              continue
                          if k ==2:
                              det_label2 = [8, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
                          if k == data_index:
                              continue
                          det_label = [x for x in det_label1 if x in det_label2]
                          #det_label = det_label2
                          if len(det_label) == 0:
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
                      #  if result_all is not None:    
                      #    for m in range(len(result_all)):
                      #       
                      #      new_result = []
                      #      
                      #      if bool(result_all[m]):  
                      #        for i in range(1,85):
                      #          labels = result_all[m].get_field('labels')
                      #          ind = labels == i
                      #          new_result.append(boxlist_nms(result_all[m][ind],0.5))
                      #      else:  
                      #            continue
                      #      new_result = cat_boxlist(new_result)

                      #      result_all[m] = new_result 
                  
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

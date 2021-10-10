import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from .inference import  make_retinanet_postprocessor
from .loss import make_retinanet_loss_evaluator
from ..anchor_generator import make_anchor_generator_retinanet
from ..utils import permute_and_flatten
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
import pdb
#def permute_and_flatten(layer, N, A, C, H, W):
#    layer = layer.view(N, -1, C, H, W)
#    layer = layer.permute(0, 3, 4, 1, 2)
#    layer = layer.reshape(N, -1, C)
#    return layer


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        self.class_num = num_classes
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                        * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE

        
        self.size = 200
        self.anchor_num = num_anchors
        f = open('semantic_vector_coco.txt','r')
        lines = f.readlines()
        sem_matrix = np.ones((80,self.size))
    #    self.ignore_labels = self.IGNORE_LABEL
        for i in range(len(lines)):
            words = lines[i].split(' ')
            words.remove(words[0])
            vec = np.array(words)
            sem_matrix[i,:] = vec
            

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_sem = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        
        self.cls_logits = nn.Conv2d(
             self.size, num_classes, kernel_size=1, stride=1
        )

        sem_matrix = torch.from_numpy(sem_matrix)
        
        sem_matrix = sem_matrix.unsqueeze(2)
        sem_matrix = sem_matrix.unsqueeze(2)

       # self.cls_logits.weight.data = sem_matrix
       # torch.nn.init.constant_(self.cls_logits.bias, 0)

       # params = self.cls_logits.parameters()
       # for param in params:
       #       param.requires_grad = False
           
#        for i in range(num_classes):
#         
#            conv = nn.Conv2d(
#            num_anchors * 400, num_anchors, kernel_size=1, stride=1,
#            )
#            sem_matrix_i = torch.from_numpy(sem_matrix[i,:])
#            sem_matrix_i = sem_matrix_i.unsqueeze(0)
#            sem_matrix_i = sem_matrix_i.unsqueeze(2)
#            sem_matrix_i = sem_matrix_i.unsqueeze(2)
#            sem_matrix_i = sem_matrix_i.repeat(num_anchors,num_anchors,1,1)
#            torch.nn.init.constant_(conv.bias, 0)
#
#            conv.weight.data = sem_matrix_i
#            params = conv.parameters()
#            for param in params:
#               param.requires_grad = False
#            self.cls_logits.append(conv)
#        self.cls_logits = nn.ModuleList(self.cls_logits)

        self.bbox_pred = nn.Conv2d(
            in_channels,  num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )
       
        # Initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_sem,
                  self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)


        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_sem.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        weight_list = []
        bias_list = []
        #weight = self.cls_logits.weight.data
        #bias = self.cls_logits.bias.data
        for feature in x:
#            for i in range(self.class_num):
#                if i == 0:
#                    outputs = self.cls_logits[i](self.cls_sem(self.cls_tower(feature)))
#                else:    
#                    outputs = torch.cat((outputs,self.cls_logits[i](self.cls_sem(self.cls_tower(feature)))),1)
             
             bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
             feature = self.cls_sem(self.cls_tower(feature))
             N = feature.size(0)
             H = feature.size(2)
             W = feature.size(3)
             
             feature = feature.reshape(feature.size(0),-1, self.size, H, W)                    
             A = feature.size(1)
             feature = feature.permute(0,2,1,3,4)
             feature = feature.reshape(N,feature.size(1), A*H, W)
             feature = self.cls_logits(feature)
             feature = feature.reshape(N,self.class_num,A,H,W)
             feature = feature.permute(0,2,1,3,4)
             feature = feature.reshape(N, A*self.class_num, H,W)
            
             #semantic_feature = self.cls_sem(feature)
#            class_logits = torch.ones(feature.size(0), self.anchor_num*self.class_num, feature.size(2),feature.size(3))
#            pdb.set_trace()
#            for i in range(self.anchor_num):
#                 
#                sem_pred = semantic_feature[:,i*400:(i+1)*400,:,:]       
#                for j in range(self.class_num):
#                     sem_vector = sem[j,:] 
#                     sem_vector = torch.from_numpy(sem_vector)
#                     
#                     sem_vector = sem_vector.unsqueeze(0)
#                     sem_vector = sem_vector.unsqueeze(2)
#                     sem_vector = sem_vector.unsqueeze(3)
#
#                     sem_vector = sem_vector.repeat(feature.size(0) , 1, feature.size(2), feature.size(3))
#                     sem_vector = sem_vector.type(torch.cuda.FloatTensor)
#                     score = torch.mean(torch.mul(sem_pred,sem_vector),1)
#                     class_logits[:,i*self.anchor_num+j,:,:] = score
             logits.append(feature)
        return logits, bbox_reg


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator_retinanet(cfg)
        head = RetinaNetHead(cfg, in_channels)
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        box_selector_test = make_retinanet_postprocessor(cfg, box_coder, is_train=False)

        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None, affinity=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
 
        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, box_cls, box_regression, targets, affinity=None):

        loss_box_cls, loss_box_reg = self.loss_evaluator(
            anchors, box_cls, box_regression, targets)
        losses = {
            "loss_retina_cls": loss_box_cls,
            "loss_retina_reg": loss_box_reg,
        }
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}


def build_retinanet(cfg, in_channels):
    return RetinaNetModule(cfg, in_channels)

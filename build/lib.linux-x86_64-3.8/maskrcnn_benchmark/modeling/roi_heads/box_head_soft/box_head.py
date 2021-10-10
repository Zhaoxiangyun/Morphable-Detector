# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
import pdb

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels,class_num=21):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels, class_num=class_num)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals,targets=None,ignore_labels=None, testing=False, previous_logits=None, get_det=False, weighting= False, targets2=None, det_labels=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
              if targets2 is None: 
                 proposals = self.loss_evaluator.subsample(proposals, targets)
              else:
                 proposals = self.loss_evaluator.subsample(proposals, targets, targets2=targets2)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        
        if testing:
             return class_logits, {}, {}
         
        if previous_logits is not None:
            labels = [1,3]
            for lab in labels:
                previous_logits[:,lab] = class_logits[:,lab]
            class_logits = previous_logits
        
        if weighting:
          domain_score =  torch.cat(proposals[0].get_field('domain_score'), proposals[1].get_field('domain_score'))
          class_logits = class_logits(domain_score > 0)
          box_regression = box_regression(domain_score > 0)
        if get_det:
            result = self.post_processor((class_logits, box_regression), proposals, labels=det_labels)
            return x, result, {}
        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}
        if not ignore_labels:
          loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
           )
        else:
          loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression], ignore_labels=ignore_labels
           )

        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels, class_num=21):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels, class_num=class_num)

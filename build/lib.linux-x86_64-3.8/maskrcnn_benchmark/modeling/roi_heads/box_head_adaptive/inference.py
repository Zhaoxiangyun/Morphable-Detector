# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
import pdb

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh

        self.score_thresh_1 = 105*[0.6]
#        low_labels = [4,6,7,8,10,40,41,42,43,59,64,65,67,69,75,76,95,96]
#        for lab in low_labels:
#            self.score_thresh_1[lab]=0.05
        self.score_thresh_1[1]=0.8
        self.score_thresh_1[2]=0.7
        self.score_thresh_1[3]=0.8
        self.score_thresh_1[4]=0.8
        self.score_thresh_1[6]=0.9
        self.score_thresh_1[7]=0.6
        self.score_thresh_1[8]=0.6
        self.score_thresh_1[11]=0.6
        self.score_thresh_1[40]=0.8
        self.score_thresh_1[41]=0.9
        self.score_thresh_1[42]=0.8
        self.score_thresh_1[44]=0.9
        self.score_thresh_1[45]=0.9
        self.score_thresh_1[59]=0.9
        self.score_thresh_1[64]=0.8
        self.score_thresh_1[65]=0.8
        self.score_thresh_1[69]=0.8
        self.score_thresh_1[71]=1.0
        self.score_thresh_1[73]=0.8
        self.score_thresh_1[84]=0.8



        self.score_thresh_2 = 105*[0.3]
        self.score_thresh_2[8] = 0.5
        low_labels = [2,6,8,11,42,65]
       # for lab in low_labels:
       #     self.score_thresh_2[lab]=0.7
        self.score_thresh_2[41]=1.0
        self.score_thresh_2[71]=1.0

        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward(self, x, boxes, labels=None):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)
        d_score = boxes[0].get_field('domain_score')
        num_per_batch = d_score.shape[0]
        start_index = 0
        end_index = 0
#        for i in range(int(class_prob.shape[0]/num_per_batch)):
#            d_score = boxes[i].get_field('domain_score')  
#            d_score = d_score.type(torch.cuda.FloatTensor)
#            end_index = start_index + d_score.shape[0]
#            for j in range(class_prob.shape[1]):
#                 class_prob[start_index:end_index,j] = class_prob[start_index:end_index,j]*d_score   
#            start_index = end_index
# TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                if labels is not None:
                   boxlist = self.filter_results(boxlist, num_classes, labels=labels)
                else: 
                    boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes, labels=None):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        if labels is None:
          inds_all = scores > self.score_thresh

        for j in range(1, num_classes):
            if labels is not None:
               if j not in labels:
                  continue
            if labels is not None:        
              inds_all = scores  > self.score_thresh_2[j]

            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)

            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            
            if labels is not None:        
              scores_after_nms = boxlist_for_class.get_field('scores')
              inds_positive = scores_after_nms  > self.score_thresh_1[j]
              inds_ignore_1 = scores_after_nms  < self.score_thresh_1[j]
              inds_ignore_2 = scores_after_nms  > self.score_thresh_2[j]
              inds_ignore = inds_ignore_1*inds_ignore_2
              result.append(boxlist_for_class[inds_positive])


                #  inds = inds_ignore[:, j].nonzero().squeeze(1)
                #  scores_j = scores[inds, j]
                #  boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
                #  boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                #  boxlist_for_class.add_field("scores", scores_j)
                #  boxlist_for_class = boxlist_nms(
                #      boxlist_for_class, self.nms
                #  )
              num_labels = len(boxlist_for_class)
            
              ignore = -1
              boxlist_for_class.add_field(
                      "labels", torch.full((num_labels,), ignore, dtype=torch.int64, device=device)
                  )
              result.append(boxlist_for_class[inds_ignore])
            else:
               result.append(boxlist_for_class)

        if len(result) == 0:
            result = {}
        else:
            result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled
    )
    return postprocessor

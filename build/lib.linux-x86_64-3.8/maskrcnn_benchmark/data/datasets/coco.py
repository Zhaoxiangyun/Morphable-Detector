# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
import pdb

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
       # f = open('test_categories.txt','w')
       # for i in range(1,201):
       #       content = self.categories[i]+'\n'
       #       f.write(content)
       # f.close()
       # pdb.set_trace()      
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
       # self.json_category_id_to_contiguous_id[1]=0
       # self.json_category_id_to_contiguous_id[2]=0
       # self.json_category_id_to_contiguous_id[3]=0
       # self.json_category_id_to_contiguous_id[4]=0
       # self.json_category_id_to_contiguous_id[5]=0
       # self.json_category_id_to_contiguous_id[6]=0
       # self.json_category_id_to_contiguous_id[7]=0
       # self.json_category_id_to_contiguous_id[8]=0
       # self.json_category_id_to_contiguous_id[9]=0
       # self.json_category_id_to_contiguous_id[10]=0
       # self.json_category_id_to_contiguous_id[11]=0

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
       
        ignore_labels = []
       # ignore_labels = [6,17,57]
       # ignore_labels = [1,2,3,4,5,6,7,9,15,16,17,18,19,20,40,57,58,59,61,63]
       # ignore_labels = list(range(2,81))
        ignore_labels = [self.contiguous_category_id_to_json_id[c] for c in ignore_labels] 
        boxes = []
        for obj in anno:
          if obj["category_id"] not in ignore_labels:
            box = obj["bbox"]
            boxes.append(box)
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        classes=[]
        for obj in anno:
          if obj["category_id"] not in ignore_labels:
            clss = obj["category_id"]
            classes.append(clss)
     #   classes = [obj["category_id"]  for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
             
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

       # if anno and "segmentation" in anno[0]:
       #     masks = [obj["segmentation"] for obj in anno]
       #     masks = SegmentationMask(masks, img.size, mode='poly')
       #     target.add_field("masks", masks)

       # if anno and "keypoints" in anno[0]:
       #     keypoints = [obj["keypoints"] for obj in anno]
       #     keypoints = PersonKeypoints(keypoints, img.size)
       #     target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        idx_name = self.id_to_img_map[idx]
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

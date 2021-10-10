import os

import torch
import torch.utils.data
from PIL import Image
import sys
import pdb
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList


class LisaDataset(torch.utils.data.Dataset):

#    CLASSES = (
#        "__background__ ",
#        "warning",
#        "stop",
#        "speedlimit",
#        "noturn",
#        "dontcare",
#    )
    CLASSES = (
        "__background__ ",
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
        "warning",
        "speedlimit",
        "noturn",
        "face",
)
#    CLASSES = (
#        "__background__ ",
#        "person",
#        "bicycle",
#        "car",
#        "motorbike",
#        "aeroplane",
#        "bus",
#        "train",
#        "truck",
#        "boat",
#        "traffic light",
#        "fire hydrant",
#        "stop",
#        "parking meter",
#        "bench",
#        "bird",
#        "cat",
#        "dog",
#        "horse",
#        "sheep",
#        "cow",
#        "elephant",
#        "bear",
#        "zebra",
#        "giraffe",
#        "backpack",
#        "umbrella",
#        "handbag",
#        "tie",
#        "suitcase",
#        "frisbee",
#        "skis",
#        "snowboard",
#        "sports ball",
#        "kite",
#        "baseball bat",
#        "baseball glove",
#        "skateboard",
#        "surfboard",
#        "tennis racket",
#        "bottle",
#        "wine glass",
#        "cup",
#        "fork",
#        "knife",
#        "spoon",
#        "bowl",
#        "banana",
#        "apple",
#        "sandwich",
#        "orange",
#        "broccoli",
#        "carrot",
#        "hot dog",
#        "pizza",
#        "donut",
#        "cake",
#        "chair",
#        "sofa",
#        "pottedplant",
#        "bed",
#        "diningtable",
#        "toilet",
#        "tvmonitor",
#        "laptop",
#        "mouse",
#        "remote",
#        "keyboard",
#        "cell phone",
#        "microwave",
#        "oven",
#        "toaster",
#        "sink",
#        "refrigerator",
#        "book",
#        "clock",
#        "vase",
#        "scissors",
#        "teddy bear",
#        "hair drier",
#        "toothbrush",
#        "dontcare",
#        "warning",
#        "speedlimit",
#        "noturn",
#        "face",
#        "coca_cola_glass_bottle",
#        "honey_bunches_of_oats_honey_roasted",
#        "nature_valley_soft_baked_oatmeal_squares_cinnamon_brown_sugar",
#        "nature_valley_sweet_and_salty_nut_almond",
#        "pringles_bbq",
#        "coffee_mate_french_vanilla",
#        "mahatma_rice",
#        "palmolive_orange", 
#        "red_bull", 
#        "pop_secret_light_butter",
#        "hunts_sauce",
#)


    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = LisaDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        index_name = self.id_to_img_map[index]
        return img, target, index_name

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            if name == "dontcare":
                continue

            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return LisaDataset.CLASSES[class_id]

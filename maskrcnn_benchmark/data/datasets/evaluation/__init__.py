from maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation
from .kitti import kitti_evaluation
from .lisa import lisa_evaluation
from .kitchen import kitchen_evaluation
from .combined import combined_evaluation
from .union import union_evaluation
from .uod import uod_evaluation
from .widerface import widerface_evaluation
from .fsod import fsod_evaluation




def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.KittiDataset):
        return kitti_evaluation(**args)
    elif isinstance(dataset, datasets.LisaDataset):
        return lisa_evaluation(**args)
    elif isinstance(dataset, datasets.KitchenDataset):
        return kitchen_evaluation(**args)
    elif isinstance(dataset, datasets.CombinedDataset):
        return combined_evaluation(**args)
    elif isinstance(dataset, datasets.UnionDataset):
        return union_evaluation(**args)
    elif isinstance(dataset, datasets.UodDataset):
        return uod_evaluation(**args)
    elif isinstance(dataset, datasets.WiderfaceDataset):
        return widerface_evaluation(**args)
    elif isinstance(dataset, datasets.FsodDataset):
        return fsod_evaluation(**args)


    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))

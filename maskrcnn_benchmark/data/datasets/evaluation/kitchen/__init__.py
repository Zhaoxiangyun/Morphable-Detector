import logging

from .kitchen_eval import do_kitchen_evaluation


def kitchen_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("kitchen evaluation doesn't support box_only, ignored.")
    logger.info("performing kitchen evaluation, ignored iou_types.")
    return do_kitchen_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )

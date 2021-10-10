import logging

from .union_eval import do_union_evaluation


def union_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("union evaluation doesn't support box_only, ignored.")
    logger.info("performing union evaluation, ignored iou_types.")
    return do_union_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )

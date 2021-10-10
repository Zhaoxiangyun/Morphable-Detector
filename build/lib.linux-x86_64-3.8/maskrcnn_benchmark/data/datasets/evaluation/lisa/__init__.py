import logging

from .lisa_eval import do_lisa_evaluation


def lisa_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("lisa evaluation doesn't support box_only, ignored.")
    logger.info("performing lisa evaluation, ignored iou_types.")
    return do_lisa_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )

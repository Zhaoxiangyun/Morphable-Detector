import logging

from .combined_eval import do_combined_evaluation


def combined_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("combined evaluation doesn't support box_only, ignored.")
    logger.info("performing combined evaluation, ignored iou_types.")
    return do_combined_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )

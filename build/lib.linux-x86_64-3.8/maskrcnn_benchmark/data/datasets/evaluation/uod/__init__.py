import logging

from .uod_eval import do_uod_evaluation


def uod_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("uod evaluation doesn't support box_only, ignored.")
    logger.info("performing uod evaluation, ignored iou_types.")
    return do_uod_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )

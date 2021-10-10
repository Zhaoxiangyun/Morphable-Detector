import logging

from .widerface_eval import do_widerface_evaluation


def widerface_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("widerface evaluation doesn't support box_only, ignored.")
    logger.info("performing widerface evaluation, ignored iou_types.")
    return do_widerface_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )

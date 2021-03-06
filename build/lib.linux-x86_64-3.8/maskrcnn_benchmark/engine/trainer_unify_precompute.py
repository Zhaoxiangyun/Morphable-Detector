# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from scipy.io import loadmat
from apex import amp
import pdb
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import numpy as np
def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader_list,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader_list[0])
    start_iter = arguments["iteration"]

    model.train()
    start_training_time = time.time()
    end = time.time()
    data_list = []
    for data_loader in data_loader_list:
       data_list.append(iter(data_loader))
    
  #  for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
    for j in range(max_iter):  
      for i in range(len(data_list)):
        data = data_list[i]
        try:
          images, targets, image_ids = next(data_list[i])
        except StopIteration:
            data_list[i] = iter(data_loader_list[i])
            images, targets, image_ids = next(data_list[i])
        if i == 1:    
            
          targets = list(targets)
          for im_i in range(len(image_ids)):
             image_size = (images.image_sizes[im_i][1],images.image_sizes[im_i][0])
             image_id = image_ids[im_i]
             box = loadmat('./boxes_coco/'+str(image_id)+'_box.mat')
             box = box['mydata']
             target_new = BoxList(box,image_size)
             labels = loadmat('./boxes_coco/'+str(image_id)+'_labels.mat')
             labels = labels['mydata']
             labels = np.squeeze(labels)
             target_new.add_field('labels',torch.tensor(labels))
             if bool(target_new): 
              try: 
                target_new.add_field('difficult', targets[im_i].get_field('difficult'))
              except:
                fields = [f for f in targets[im_i].fields() if f != 'masks']
                targets[im_i] = targets[im_i].copy_with_fields(fields)
              try: 
                  targets[im_i] = cat_boxlist((targets[im_i],target_new))             
                  
              except:
                      continue
          targets = tuple(targets)   
        iteration = j+1
        if any(len(target) < 1 for target in targets):
           logger.error("Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
           if iteration % checkpoint_period == 0:
             checkpointer.save("model_{:07d}".format(iteration), **arguments)
           if iteration == max_iter:
              checkpointer.save("model_final", **arguments)

           continue
        data_time = time.time() - end
        arguments["iteration"] = iteration
        
        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        loss_dict = model(images, targets, i+1, iteration=iteration)
        losses = sum(loss for loss in loss_dict.values()) + 0 * sum(p.sum() for p in model.parameters())


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
      if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
      if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

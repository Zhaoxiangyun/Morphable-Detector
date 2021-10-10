# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from apex import amp
import pdb
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
    ignore_label_list = None,
    pseudo = False,
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
    if pseudo:
      face = True  
      if not face:  
        pred_list = []   
        pred_dir = "/home/mai/xzhao/maskrcnn-benchmark/output/pseudo_label/lisa_voc_sun_coco_all_seperate/inference/voc_0712_trainval/predictions.pth"
        prediction = torch.load(pred_dir)
        pred_list.append(prediction)
        pred_dir = "/home/mai/xzhao/maskrcnn-benchmark/output/pseudo_label/lisa_voc_sun_coco_all_seperate/inference/coco_2017_train/predictions.pth"
        prediction = torch.load(pred_dir)
        pred_list.append(prediction)
        pred_dir = "/home/mai/xzhao/maskrcnn-benchmark/output/pseudo_label/lisa_voc_sun_coco_all_seperate/inference/lisa_train/predictions.pth"
        prediction = torch.load(pred_dir)
        pred_list.append(prediction)
        pred_dir = "/home/mai/xzhao/maskrcnn-benchmark/output/pseudo_label/lisa_voc_sun_coco_all_seperate/inference/kitti_train/predictions.pth"
        prediction = torch.load(pred_dir)
        pred_list.append(prediction)
        pred_dir = "/home/mai/xzhao/maskrcnn-benchmark/output/pseudo_label/lisa_voc_sun_coco_all_seperate/inference/sun_train/predictions.pth"
        prediction = torch.load(pred_dir)
        pred_list.append(prediction)
        print('load data done!') 
      else:
        pred_list = []   
        pred_dir = "/home/mai/xzhao/maskrcnn-benchmark/output/pseudo_label/coco_widerface_human_seperately/inference/voc_0712_trainval_1/predictions.pth"
        prediction = torch.load(pred_dir)
        pred_list.append(prediction)
        pred_dir = "/home/mai/xzhao/maskrcnn-benchmark/output/pseudo_label/coco_widerface_human_seperately/inference/coco_2017_train/predictions.pth"
        prediction = torch.load(pred_dir)
        pred_list.append(prediction)
        pred_dir = "/home/mai/xzhao/maskrcnn-benchmark/output/pseudo_label/coco_widerface_human_seperately/inference/widerface_train/predictions.pth"
        prediction = torch.load(pred_dir)
        pred_list.append(prediction)
        print('load widerface image data done!') 

  #  for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
    for j in range(max_iter):
      
      iteration = j+1
      arguments["iteration"] = iteration
      for i in range(len(data_list)):
        data = data_list[i]
        #if i == 0:
        #    continue
       # if ignore_label_list is not None:
       #  if i in ignore_label_list:
       #     continue

       # if any(len(target) < 1 for target in targets):
       #    logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )

       #    if iteration % checkpoint_period == 0:
       #       checkpointer.save("model_{:07d}".format(iteration), **arguments)
       #    if iteration == max_iter:
       #       checkpointer.save("model_final", **arguments)
       #    

       #    continue
        data_time = time.time() - end
        try:
            
          images, targets, ids = next(data_list[i])
          targets = [target.to(device) for target in targets]

          if i == 0:
             keep_labels = list(range(1,41))
          else:
             keep_labels = list(range(41,81))
          for ti in range(len(targets)):
              target = targets[ti]
              labels = target.get_field('labels')
              new_target = []
              for lab in keep_labels:
                  inds = labels == lab
                  new_target.append(target[inds])
              
              new_target =cat_boxlist(new_target)  
              targets[ti] = new_target

          while(any(len(target) < 1 for target in targets)):
            try:
                 images, targets, ids = next(data_list[i])
                 targets = [target.to(device) for target in targets]

                 for ti in range(len(targets)):
                       target = targets[ti]
                       labels = target.get_field('labels')
                       new_target = []
                       for lab in keep_labels:
                             inds = labels == lab
                             new_target.append(target[inds])
              
                       new_target =cat_boxlist(new_target)  
                       targets[ti] = new_target
                
            except:
                 data_list[i] = iter(data_loader_list[i])
                 images, targets, ids = next(data_list[i])
                 targets = [target.to(device) for target in targets]

                 for ti in range(len(targets)):
                       target = targets[ti]
                       labels = target.get_field('labels')
                       new_target = []
                       for lab in keep_labels:
                             inds = labels == lab
                             new_target.append(target[inds])
              
                       new_target =cat_boxlist(new_target)  
                       targets[ti] = new_target

        except StopIteration:
            data_list[i] = iter(data_loader_list[i])
            images, targets, ids = next(data_list[i])
            while(any(len(target) < 1 for target in targets)):
               try:  
                  images, targets, ids = next(data_list[i])
               except:
                 data_list[i] = iter(data_loader_list[i])
                 images, targets, ids = next(data_list[i])
       

        scheduler.step()
        images = images.to(device)
        #try:
        if pseudo:
         targets2 = []
         for img_id in ids:
             targets2.append(pred_list[i][img_id].to(device))
         if pseudo:
           print(i)
        else: 
            targets2 = None
        #try:
        loss_dict = model(images, targets, targets2, i+1, iteration=iteration)
       # except:
       #     pdb.set_trace()
       #     continue
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

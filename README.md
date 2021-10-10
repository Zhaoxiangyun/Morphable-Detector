## Morphable Detector for Object Detection on Demand
This is a PyTorch implementation of the [Morphable Detector]() paper.
```
@inproceedings{zhaomorph,
  author  = {Xiangyun Zhao, Xu Zou, Ying Wu},
  title   = {Morphable Detector for Object Detection on Demand},
  booktitle = {ICCV},
  Year  = {2021}
}
```


### Install

First, [install PyTorch](https://pytorch.org/get-started/locally/) and torchvision. We have tested on version of 1.8.0 with CUDA 11.0, but the other versions should also be working.


Our code is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), so you should install all dependencies. 

### Data Preparation

Download large scale few detection dataset [here](https://github.com/fanq15/Few-Shot-Object-Detection-Dataset) and covert the data into COCO dataset format. The file structure should look like:

```
  $ tree data
  dataset
  ├──fsod
      ├── annototation
      │   
      ├── images
```

### Training (EM-like approach)
All of the models are using COCO dataset to pretrain the model. The COCO dataset pretrained model can be found here.

We first initialize the prototypes using semantic vectors, then train the network run:
```
export NGPUS=2
RND_PORT=`shuf -i 4000-7999 -n 1`

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port $RND_PORT --nproc_per_node=$NGPUS ./tools/train_sem_net.py \
--config-file "./configs/fsod/e2e_faster_rcnn_R_50_FPN_1x.yaml"  SEM_DIR "." OUTPUT_DIR "YOUR_OUTPUT_PATH" \
MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN  2000 SOLVER.IMS_PER_BATCH 4 SOLVER.MAX_ITER 70000 \
SOLVER.STEPS "(50000,70000)" SOLVER.CHECKPOINT_PERIOD 10000 \
SOLVER.BASE_LR 0.002  
```

Then, we update the feature vectors of the training samples on the trained network, run:
```
export NGPUS=2
RND_PORT=`shuf -i 4000-7999 -n 1`

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port $RND_PORT --nproc_per_node=$NGPUS \
./tools/train_sem_net.py --config-file "./configs/fsod/e2e_faster_rcnn_R_50_FPN_1x.yaml"  \ 
FEATURE_DIR "features" OUTPUT_DIR "WHERE_YOU_SAVE_YOUR_MODEL" \
FEATURE_SIZE 200 SEM_DIR "visual_sem.txt" FEATURE True \
MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN  2000 \
SOLVER.IMS_PER_BATCH 4 SOLVER.MAX_ITER 80000 \
SOLVER.CHECKPOINT_PERIOD 10000000
```

To compute the mean vectors and update the prototypes, run

```
cd features

python mean_features.py
python update_prototype.py

```

To train the network using the updated prototypes, run
```
export NGPUS=2
RND_PORT=`shuf -i 4000-7999 -n 1`

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port $RND_PORT --nproc_per_node=$NGPUS \
./tools/train_sem_net.py --config-file "./configs/fsod/e2e_faster_rcnn_R_50_FPN_1x.yaml"  \
SEM_DIR "PATH_WHERE_YOU_SAVE_THE_PROTOTYPES" VISUAL True OUTPUT_DIR "./output/fsod/resnet50/language_visual_ver1_iter1_wo" \ 
MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN  2000 SOLVER.IMS_PER_BATCH 4 \
SOLVER.MAX_ITER 70000 SOLVER.STEPS "(50000,80000)" \
SOLVER.CHECKPOINT_PERIOD 10000 \
SOLVER.BASE_LR 0.002 

```

### Tests

After the model is trained, we randomly sample 5 samples for each novel category from the test data and use the mean feature vectors for the 5 samples as the prototype for that categpry. 
To extract the features for test data, run 

We provide the mean feature vectors for 5 samples extracted by the pretrained model here.


### Models

Our pre-trained ResNet-50 models can be downloaded as following:

| name | iterations | AP | AP^{0.5} | model |
| :---: | :---: | :---: | :---: | :---: |
| MD (initial training) | 70,000 |   | | [download]() |
| MD | 70,000 |   | 37.9 | [download]() |


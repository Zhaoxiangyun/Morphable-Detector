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
We follow [FSOD Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Few-Shot_Object_Detection_With_Attention-RPN_and_Multi-Relation_Detector_CVPR_2020_paper.pdf), we pretrain the model using COCO dataset for 200,000 iterations. So, you can download the model [here](), and use it to initilize the network. 

We first initialize the prototypes using semantic vectors, then train the network run:
```
export NGPUS=2
RND_PORT=`shuf -i 4000-7999 -n 1`

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port $RND_PORT --nproc_per_node=$NGPUS ./tools/train_sem_net.py \
--config-file "./configs/fsod/e2e_faster_rcnn_R_50_FPN_1x.yaml"  OUTPUT_DIR "YOUR_OUTPUT_PATH" \
MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN  2000 SOLVER.IMS_PER_BATCH 4 SOLVER.MAX_ITER 270000 \
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
FEATURE_SIZE 200 SEM_DIR "visual_sem.txt" GET_FEATURE True \
MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN  2000 \
SOLVER.IMS_PER_BATCH 4 SOLVER.MAX_ITER 80000 \
SOLVER.CHECKPOINT_PERIOD 10000000
```

To compute the mean vectors and update the prototypes, run

```
cd features

python mean_features.py FEATURE_FILE MEAN_FEATURE_FILE
python update_prototype.py MEAN_FEATURE_FILE

```

To train the network using the updated prototypes, run
```
export NGPUS=2
RND_PORT=`shuf -i 4000-7999 -n 1`

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port $RND_PORT --nproc_per_node=$NGPUS \
./tools/train_sem_net.py --config-file "./configs/fsod/e2e_faster_rcnn_R_50_FPN_1x.yaml"  \
SEM_DIR "PATH_WHERE_YOU_SAVE_THE_PROTOTYPES" VISUAL True OUTPUT_DIR "WHERE_YOU_SAVE_YOUR_MODEL" \ 
MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN  2000 SOLVER.IMS_PER_BATCH 4 \
SOLVER.MAX_ITER 70000 SOLVER.STEPS "(50000,80000)" \
SOLVER.CHECKPOINT_PERIOD 10000 \
SOLVER.BASE_LR 0.002 

```

### Tests

After the model is trained, we randomly sample 5 samples for each novel category from the test data and use the mean feature vectors for the 5 samples as the prototype for that categpry. 
To extract the features for test data, run 
```
export NGPUS=2
RND_PORT=`shuf -i 4000-7999 -n 1`

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port $RND_PORT --nproc_per_node=$NGPUS \
./tools/train_sem_net.py --config-file "./configs/fsod/e2e_faster_rcnn_R_50_FPN_1x.yaml"  \ 
FEATURE_DIR "features" OUTPUT_DIR "WHERE_YOU_SAVE_YOUR_MODEL" \
FEATURE_SIZE 200 SEM_DIR "visual_sem.txt" GET_FEATURE True \
MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN  2000 \
SOLVER.IMS_PER_BATCH 4 SOLVER.MAX_ITER 80000 \
SOLVER.CHECKPOINT_PERIOD 10000000
```
To compute the prototype for each class, run

```
cd features

python mean_features.py FEATURE_FILE MEAN_FEATURE_FILE

```

Then run test,


```
export NGPUS=2
RND_PORT=`shuf -i 4000-7999 -n 1`

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port $RND_PORT --nproc_per_node=$NGPUS ./tools/test_sem_net.py --config-file "./configs/fsod/e2e_faster_rcnn_R_50_FPN_1x.yaml" SEM_DIR WHERE_YOU_SAVE_THE_PROTOTYPES VISUAL True OUTPUT_DIR WHERE_YOU_SAVE_THE_MODEL MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 FEATURE_SIZE 200 MODEL.ROI_BOX_HEAD.NUM_CLASSES 201 TEST_SCALE 0.8

```


### Models

Our pre-trained ResNet-50 models can be downloaded as following:

| name | iterations | AP | AP^{0.5} | model | Mean Features |
| :---: | :---: | :---: | :---: | :---: | :---:|
| MD | 70,000 | 22.2  | 37.9 | [download](https://drive.google.com/file/d/1M78AjPtZzOVrzKYZy1FV-e3mUpJomxpa/view?usp=sharing) | [download]() |


| name | iterations | AP | AP^{0.5} | Mean Features |
| :---: | :---: | :---: | :---: | :---: |
| MD 1-shot | 70,000 |   |  | [download](https://drive.google.com/file/d/1M78AjPtZzOVrzKYZy1FV-e3mUpJomxpa/view?usp=sharing) |
| MD 3-shot | 70,000 |   |  | [download](https://drive.google.com/file/d/1M78AjPtZzOVrzKYZy1FV-e3mUpJomxpa/view?usp=sharing) |
| MD 5-shot | 70,000 | 22.2  | 37.9 | [download](https://drive.google.com/file/d/1M78AjPtZzOVrzKYZy1FV-e3mUpJomxpa/view?usp=sharing) |






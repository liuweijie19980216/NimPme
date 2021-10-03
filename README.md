# Novel Instances Mining with Pseudo-Margin Evaluation for Few-Shot Object Detection (NimPme)

The official implementation of **Novel Instances Mining with Pseudo-Margin Evaluation for Few-Shot Object Detection**
<p align="center">
<img src="https://github.com/liuweijie19980216/NimPme/blob/master/imgs/fig2.png" width="800px" alt="teaser">
</p>

The code is built on [TFA](https://github.com/ucbdrive/few-shot-object-detection)

**Requirements**

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.4
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* CUDA 10.0, 10.1, 10.2
* GCC >= 4.9


## Getting Started

### Evaluation with pre-trainied 10-shot final detecor
we provide the [pre-trainied 10-shot final detecor](https://www.dropbox.com/s/4mkc8890f0944pm/model_final.pth?dl=0)
```angular2html
python3 -m tools.test_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml \
        --eval-only
```

### Training & Evaluation in Command Line

To train a base detector, run
```angular2html
python3 -m tools.train_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml
```

fine-tune the detector with novel set
```angular2html
python3 -m tools.ckpt_surgery \
        --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth \
        --method randinit \
        --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all
        --coco
```
```angular2html
python3 -m tools.train_net --num-gpus 1 \
        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
        --opts MODEL.WEIGHTS $WEIGHTS_PATH
```
fine-tune the detector with pseudo set
     
```
python3 -m tools.genarate_pseudo --num-gpus 1
python3 -m tools.train_feature --num-gpus 1   
```


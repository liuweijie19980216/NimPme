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


## Models
We provide a set of benchmark results and pre-trained models available for download in [MODEL_ZOO.md](docs/MODEL_ZOO.md).


## Getting Started

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from
  [model zoo](fsdet/model_zoo/model_zoo.py),
  for example, `COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml`.
2. We provide `demo.py` that is able to run builtin standard models. Run it with:
```
python3 -m demo.demo --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS fsdet://coco/tfa_cos_1shot/model_final.pth
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.

### Training & Evaluation in Command Line

To train a model, run
```angular2html
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml
```

To evaluate the trained models, run
```angular2html
python3 -m tools.test_net --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
        --eval-only
```

For more detailed instructions on the training procedure of TFA, see [TRAIN_INST.md](docs/TRAIN_INST.md).

### Multiple Runs

For ease of training and evaluation over multiple runs, we provided several helpful scripts in `tools/`.

You can use `tools/run_experiments.py` to do the training and evaluation. For example, to experiment on 30 seeds of the first split of PascalVOC on all shots, run
```angular2html
python3 -m tools.run_experiments --num-gpus 8 \
        --shots 1 2 3 5 10 --seeds 0 30 --split 1
```

After training and evaluation, you can use `tools/aggregate_seeds.py` to aggregate the results over all the seeds to obtain one set of numbers. To aggregate the 3-shot results of the above command, run
```angular2html
python3 -m tools.aggregate_seeds --shots 3 --seeds 30 --split 1 \
        --print --plot
```

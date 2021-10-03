"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in FsDet.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import launch
from fsdet.evaluation import (
    COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, verify_results)
import contextlib
import os
from torch.nn.parallel import DistributedDataParallel
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.evaluation import verify_results
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.utils import comm
from fsdet.data.build import build_pseudo_train_loader
import contextlib
import logging
import numpy as np
import time
import weakref
import torch

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage

try:
    _nullcontext = contextlib.nullcontext  # python 3.7+
except AttributeError:

    @contextlib.contextmanager
    def _nullcontext(enter_result=None):
        yield enter_result


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder)
            )
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg, dataset_batch=1):
        """
        Returns:
            iterable

        It now calls :func:`fsdet.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_pseudo_train_loader(cfg, dataset_batch=dataset_batch)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """

        try:
            data = next(self._data_loader_iter)
        except:
            if self.max_dataset_batch > 1:
                self.dataset_batch += 1
                if self.dataset_batch > self.max_dataset_batch:
                    self.dataset_batch = 1
                self.data_loader = self.build_train_loader(self.cfg, dataset_batch=self.dataset_batch)
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """

        loss_dict = self.model(data)
        # if self.iter <= 500:
        #     self.cls_alpha = 1
        # elif self.iter > 500 and (self.iter % 100 == 0):
        #     iter_step = (self.iter - 500) / 100
        #     self.cls_alpha = 1 * (0.5 ** iter_step)
        # loss_dict['loss_cls'] = loss_dict['loss_cls'] * self.cls_alpha
        # if self.iter % 500 == 0:
        #     print('weight of cls_loss is:')
        #     print(self.cls_alpha)
        losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(
            torch.cuda.Stream()
        ) if losses.device.type == "cuda" else _nullcontext():
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.max_dataset_batch = 1
    trainer.dataset_batch = 1
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus = 1
    args.config_file = '/home/liuwj/Repository/few-shot-object-detection/' \
                       'configs/COCO-detection/feature_ft_pseudo_10shot.yaml'



    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

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
import torch
from torch import nn
from detectron2.data.build import get_detection_dataset_dicts
from fsdet.modeling.roi_heads import build_roi_heads
from detectron2.structures import Instances, Boxes
import logging
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2 import utils
# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import launch
from fsdet.evaluation import (
    COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, verify_results)
import contextlib
import logging
import numpy as np
import time
import weakref
import torch
from detectron2.layers import batched_nms, cat
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

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data, feature_extract=True)
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


def feature_extract_gt(cfg, model, data_loader):
    data_loader_iter = iter(data_loader)
    num_dataset = len(data_loader.dataset.dataset)
    batch_size = data_loader.batch_size
    epoch_size = num_dataset // batch_size
    all_box_features, all_box_labels = [], []
    class_features = [[] for i in range(80)]
    with EventStorage() as storage:
        for iter_id in range(epoch_size):
            if iter_id % 10 == 0:
                print(f"save {iter_id * batch_size}-th images")
            batched_inputs = next(data_loader_iter)
            images = model.preprocess_image(batched_inputs)
            if "instances" in batched_inputs[0]:
                gt_instances = [
                    x["instances"].to(model.device) for x in batched_inputs
                ]
            elif "targets" in batched_inputs[0]:
                gt_instances = [
                    x["targets"].to(model.device) for x in batched_inputs
                ]
            else:
                gt_instances = None
            features = model.backbone(images.tensor)

            if model.proposal_generator:
                proposals, proposal_losses = model.proposal_generator(
                    images, features, gt_instances
                )
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(model.device) for x in batched_inputs
                ]
            box_features, box_labels = model.roi_heads(
                images, features, proposals, gt_instances, extractor=True
            )
            gt_classes = cat([p.gt_classes for p in box_labels], dim=0)
            for box_id, cls in enumerate(gt_classes):
                cls = cls.item()
                if cls == 80:
                    continue
                box_feat = box_features[box_id].cpu()
                class_features[cls].append(box_feat)

    # class_features_mean = []
    # for cls in class_features:
    #     if len(cls) > 0:
    #         cls_fea = sum(cls) / len(cls)
    #         class_features_mean.append(cls_fea)
    #     else:
    #         class_features_mean.append(torch.zeros(1024))

    torch.save(class_features, 'class_features.pt')



def feature_extract(cfg, model, data_loader):
    data_loader_iter = iter(data_loader)
    num_dataset = len(data_loader.dataset.dataset)
    batch_size = data_loader.batch_size
    epoch_size = num_dataset // batch_size
    all_box_features, all_box_labels = [], []
    file_batch_id = 0

    with EventStorage() as storage:
        for iter_id in range(epoch_size):
            if iter_id % 10 == 0:
                print(f"save {iter_id * batch_size}-th images")
            batched_inputs = next(data_loader_iter)
            images = model.preprocess_image(batched_inputs).to(model.device)
            if "instances" in batched_inputs[0]:
                gt_instances = [
                    x["instances"].to(model.device) for x in batched_inputs
                ]
            elif "targets" in batched_inputs[0]:
                gt_instances = [
                    x["targets"].to(model.device) for x in batched_inputs
                ]
            else:
                gt_instances = None

            if 'is_pseudo' in batched_inputs[0]:
                is_pseudos = [
                    x["is_pseudo"] for x in batched_inputs
                ]
            for image_id, gt_insta in enumerate(gt_instances):
                is_p = [is_pseudos[image_id]] * len(gt_insta)
                gt_insta.set('is_pseudo', is_p)
            features = model.backbone(images.tensor)
            if model.proposal_generator:
                proposals, proposal_losses = model.proposal_generator(
                    images, features, gt_instances
                )
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(model.device) for x in batched_inputs
                ]
            box_features, box_labels = model.roi_heads(
                images, features, proposals, gt_instances, extractor=True
            )
            del features
            box_features = box_features.to('cpu')

            box_features_split = box_features.view(cfg.SOLVER.IMS_PER_BATCH, -1, 1024).to('cpu')

            for img_id, fea in enumerate(box_features_split):
                all_box_features.append(fea)
                label = box_labels[img_id].to('cpu')
                all_box_labels.append(label)

            if iter_id > 0 and iter_id * batch_size % 5000 == 0:
                file_batch_id += 1
                torch.save(all_box_features, '/media/liuwj/D/liuweijie/liuweijie/few-shot-object-detection/features/'
                                             f'10shot_thres07_novelset/all_box_features_batch{file_batch_id}.pt')
                torch.save(all_box_labels, '/media/liuwj/D/liuweijie/liuweijie/few-shot-object-detection/features/'
                                           f'10shot_thres07_novelset/all_box_labels_batch{file_batch_id}.pt')



                print(f'save features and labels batch{file_batch_id}')
                all_box_features, all_box_labels = [], []

        file_batch_id += 1
        torch.save(all_box_features,
                   '/media/liuwj/D/liuweijie/liuweijie/few-shot-object-detection/features/'
                   f'10shot_thres07_novelset/all_box_features_batch{file_batch_id}.pt')
        torch.save(all_box_labels,
                   '/media/liuwj/D/liuweijie/liuweijie/few-shot-object-detection/features/'
                   f'10shot_thres07_novelset/all_box_labels_batch{file_batch_id}.pt')
        print(f'save features and labels batch{file_batch_id}')


def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    model = trainer.model
    data_loader = trainer.data_loader
    feature_extract(cfg, model, data_loader)
    #feature_extract_gt(cfg, model, data_loader)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus = 1
    args.config_file = 'configs/COCO-detection/finetune_pseudo_10shot.yaml'


    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

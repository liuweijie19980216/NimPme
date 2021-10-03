import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch
import numpy as np
import cv2
from detectron2 import utils
from detectron2.utils.comm import is_main_process
from detectron2.structures import pairwise_iou, BoxMode, Boxes
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
import json


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    vis = False
    if vis:
        anno_path = '/home/liuwj/Repository/few-shot-object-detection/datasets/cocosplit/datasplit/trainvalno5k.json'
        trainval_coco = COCO(anno_path)
        metadata = MetadataCatalog.get('coco_trainval_all_10shot')
        id_map = metadata.get("thing_dataset_id_to_contiguous_id")
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            if vis:

                #gt_boxes_xyxy = BoxMode.convert(gt_boxes.tensor.numpy(), from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYXY_ABS)
                prediction = outputs[0]['instances'].to('cpu')
                pred_boxes = prediction.get('pred_boxes')
                scores = prediction.get('scores')
                pred_classes = prediction.get('pred_classes')

                keep_thres = scores > 0.5
                pred_boxes_thres = pred_boxes[keep_thres]
                pred_classes_thres = pred_classes[keep_thres]
                scores_thres = scores[keep_thres]
                pred_cls_names, pred_cls_dataset = [], []
                for i in range(len(scores_thres)):
                    cls = _convert_contiguous_to_dataset_id(pred_classes_thres[i], id_map)
                    pred_cls_dataset.append(cls)
                    pred_cls_names.append(trainval_coco.loadCats(cls)[0]['name'])
                img_name = inputs[0]['file_name'].split('/')[-1]
                pred_img = _draw_box(inputs[0]['file_name'], pred_boxes_thres, pred_cls_names, pred_cls_dataset)
                out_path = '/home/liuwj/Repository/few-shot-object-detection/pred_vis/'+img_name
                cv2.imwrite(out_path, pred_img)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def pseudo_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    anno_path = '/home/liuwj/Repository/few-shot-object-detection/datasets/cocosplit/datasplit/trainvalno5k.json'
    trainval_coco = COCO(anno_path)
    metadata = MetadataCatalog.get('coco_trainval_all_10shot')
    id_map = metadata.get("thing_dataset_id_to_contiguous_id")
    base_classes = [
        8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 84, 85, 86, 87, 88, 89, 90,
    ]
    novel_classes = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21,
                     44, 62, 63, 64, 67, 72]
    pseudo_labels = {'images': [],
                     'annotations': []}
    pred_num_all = 0
    correct_num_all = 0
    pseudo_anno_id = 0

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            # if idx == 100:
            #     break
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()

            outputs = model(inputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            pseudo_boxes, pseudo_classes, correct_num, pred_num = \
                find_pseudo(inputs, outputs, id_map, base_classes, novel_classes, trainval_coco)
            if pred_num > 0:
                correct_num_all += correct_num
                pred_num_all += pred_num
                print("precision of novel boxes: {}/{} {:.2f}%" .format(
                    correct_num_all, pred_num_all, correct_num_all/pred_num_all*100))

                image_id = inputs[0]['image_id']
                img_info = trainval_coco.loadImgs(image_id)
                anno_ids = trainval_coco.getAnnIds(image_id)

                gt_anno = trainval_coco.loadAnns(anno_ids)

                for anno in gt_anno:
                    if anno['category_id'] in base_classes:
                        pseudo_labels['annotations'].append(anno)

                for pseudo_id in range(len(pseudo_boxes)):
                    pseudo_anno_id += 1
                    box_p = pseudo_boxes[pseudo_id].tensor.numpy().tolist()[0]
                    cls_p = pseudo_classes[pseudo_id]
                    annotation_p = {"segmentation": [],
                                      'area': 0,
                                      "iscrowd": 0,
                                      "image_id": image_id,
                                      "bbox": box_p,
                                      "category_id": cls_p,
                                      'id': pseudo_anno_id,
                                      "clean_bbox": []}
                    pseudo_labels['annotations'].append(annotation_p)
                pseudo_labels['images'].append(img_info[0])

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

        json_str = json.dumps(pseudo_labels, ensure_ascii=False)
        coco_path = 'pseudo_10shot_thres06.json'
        with open(coco_path, 'w') as f:
            f.write(json_str)

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def _draw_box(image_path, boxes, class_name, cls_dataset):
    base_classes = [
        8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 84, 85, 86, 87, 88, 89, 90,
    ]
    novel_classes = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21,
                     44, 62, 63, 64, 67, 72]
    image = cv2.imread(image_path)
    boxes = np.int32(boxes.tensor.numpy())

    for box_id, box in enumerate(boxes):
        if cls_dataset[box_id] in base_classes:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=color)
        text_size = cv2.getTextSize('%s' % (class_name[box_id] + ' '), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        point = (box[0] + text_size[0][0], box[1] + text_size[0][1] + text_size[1])
        cv2.rectangle(image, box[0:2], point, color, -1)
        cv2.putText(image, '%s' % (class_name[box_id] + ' '), (box[0], box[1] + 10), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (0, 0, 0), thickness=1)
    return image

def _convert_contiguous_to_dataset_id(contiguous_id, id_map):
    for dataset_key, contiguous_value in id_map.items():
        if contiguous_id == contiguous_value:
            return dataset_key

def find_pseudo(inputs, outputs, id_map, base_classes, novel_classes, trainval_coco, thres=0.6):

    pred_num, correct_num = 0, 0
    file_name = inputs[0]['file_name']
    annos = inputs[0]['instances_ori']
    gt_boxes = annos.get('gt_boxes')
    gt_classes = annos.get('gt_classes')
    # gt_boxes_xyxy = BoxMode.convert(gt_boxes.tensor.numpy(), from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYXY_ABS)
    prediction = outputs[0]['instances'].to('cpu')
    pred_boxes = prediction.get('pred_boxes')
    scores = prediction.get('scores')
    pred_classes = prediction.get('pred_classes')

    # 滤除阈值低于thres的预测框

    # pred_boxes_thres, pred_classes_thres, scores_thres = [], [], []
    # for pred_id in range(len(pred_classes)):
    #     pred_cls = pred_classes[pred_id]
    #     pred_box = pred_boxes[pred_id]
    #     score = scores[pred_id]
    #     pred_cls_dataset = _convert_contiguous_to_dataset_id(pred_cls, id_map)
    #     if pred_cls_dataset in novel_classes:
    #         thres = class_thres_map[str(pred_cls_dataset)]
    #         if score >= thres:
    #             pred_boxes_thres.append(pred_box)
    #             pred_classes_thres.append(pred_cls)
    #             scores_thres.append(score)
    #
    # pred_boxes_thres = Boxes.cat(pred_boxes_thres)

    keep_thres = scores > thres
    pred_boxes_thres = pred_boxes[keep_thres]
    pred_classes_thres = pred_classes[keep_thres]
    scores_thres = scores[keep_thres]

    # 给预测框匹配一个标签
    pred_gt_ious = pairwise_iou(pred_boxes_thres, gt_boxes)
    pseudo_boxes, pseudo_classes = [], []  # dataset classes id
    if len(pred_gt_ious) > 0:
        max_ious, max_ids = pred_gt_ious.max(dim=1)
        for pred_id, gt_id in enumerate(max_ids):
            gt_id = gt_id.item()
            pred_score = scores_thres[pred_id]
            pred_box = pred_boxes_thres[pred_id]
            pred_cls = pred_classes_thres[pred_id]
            pred_cls_dataset = _convert_contiguous_to_dataset_id(pred_cls, id_map)
            pred_cls_name = trainval_coco.loadCats(pred_cls_dataset)[0]['name']
            if pred_cls_dataset in novel_classes:
                pseudo_boxes.append(pred_box)
                pseudo_classes.append(pred_cls_dataset)
                pred_num += 1
                #gt_box = gt_boxes[gt_id]
                gt_cls = gt_classes[gt_id]
                gt_cls_dataset = _convert_contiguous_to_dataset_id(gt_cls, id_map)
                #gt_cls_name = trainval_coco.loadCats(gt_cls_dataset)[0]['name']
                # pred_img = _draw_box(file_name, pred_box, pred_cls_name, pred_score)
                # cv2.imwrite('pred.jpg', pred_img)
                # gt_img = _draw_box(file_name, gt_box, gt_cls_name)
                # cv2.imwrite('gt.jpg', gt_img)
                if pred_cls_dataset == gt_cls_dataset:
                    correct_num += 1
        return pseudo_boxes, pseudo_classes, correct_num, pred_num

    else:
        return [], [], correct_num, pred_num


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

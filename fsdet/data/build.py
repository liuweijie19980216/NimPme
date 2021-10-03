from detectron2.data.build import get_detection_dataset_dicts, trivial_batch_collator
import torch.utils.data
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from fsdet.data.dataset_pseudo_mapper import DatasetPseudoMapper
from detectron2.data.samplers import InferenceSampler
import argparse
import logging
import os
from collections import OrderedDict
import torch
import logging
import torch.utils.data
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_batch_data_loader
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler


def build_detection_pseudo_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetPseudoMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def build_pseudo_train_loader(cfg, dataset_batch=1):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    feature_dataset = FeatureDataset(dataset_batch=dataset_batch)
    feature_dataloader = torch.utils.data.DataLoader(feature_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
                                                     drop_last=True)
    return feature_dataloader



class FeatureDataset(torch.utils.data.Dataset):  # 创类：MyDataset,继承torch.utils.data.Dataset
    def __init__(self, dataset_batch=1):
        super(FeatureDataset, self).__init__()
        novel_classes = [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
        root_path = '/home/liuwj/Repository/few-shot-object-detection/10shot_thres07_novelset'
        print(f'load feature dataset batch: {dataset_batch}')
        feature_name = f'all_box_features_batch{dataset_batch}.pt'
        label_name = f'all_box_labels_batch{dataset_batch}.pt'
        features = torch.load(os.path.join(root_path, feature_name))
        annos = torch.load(os.path.join(root_path, label_name))

        anno_dicts = []

        for anno_id, anno in enumerate(annos):
            anno_dict = anno.get_fields()
            anno_dict['gt_boxes'] = anno_dict['gt_boxes'].tensor
            anno_dict['proposal_boxes'] = anno_dict['proposal_boxes'].tensor
            anno_dict['image_size'] = torch.tensor(anno.image_size)
            anno_dict['is_pseudo'] = torch.tensor(anno_dict['is_pseudo'])
            anno_dicts.append(anno_dict)

        self.features = features
        self.annos = anno_dicts

    def __getitem__(self, index):  # 按照索引读取每个元素的具体内容
        feature = self.features[index]
        anno = self.annos[index]
        return feature, anno  # return回哪些内容，在训练时循环读取每个batch，就能获得哪些内容

    def __len__(self):  # 它返回的是数据集的长度，必须有
        return len(self.features)


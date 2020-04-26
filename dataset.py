from __future__ import absolute_import, division, print_function

from fileloader import DirLoader, MultiDirLoader
from datautil import ZscoreTransform
from datautil import sliding_window_wrapper, batch_and_mask

import os

import numpy as np
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf


SAME_GRAPH = "same"
DIFF_GRAPH = "diff"


def build_dataloader(config):
    scale, shift = config.preprocess_scale, config.preprocess_shift
    assert (scale is None) == (shift is None)

    if scale is not None:
        scale = np.array([float(x) for x in scale])
        shift = np.array([float(x) for x in shift])
        assert scale.shape == shift.shape
        assert not config.zscore
        transform = ZscoreTransform(mean=shift, stddev=scale)
    else:
        transform = None

    train_dir = os.path.join(config.dataset, "train")
    eval_dir = config.eval_dataset or \
        os.path.join(config.dataset, "test" if config.test else "val")

    if config.dataset_type == SAME_GRAPH:
        train_loader = DirLoader(
            dir=train_dir,
            zscore=config.zscore, transform=transform
        )
        eval_loader = DirLoader(
            dir=eval_dir,
            zscore=False, transform=train_loader.transform
        )
        train_dataset = SameGraphDataset(
            loader=train_loader,
            win_size=config.win_size, stride=config.stride
        )
        eval_dataset = SameGraphDataset(
            loader=eval_loader,
            win_size=config.win_size, stride=config.stride
        )
    elif config.dataset_type == DIFF_GRAPH:
        train_loader = MultiDirLoader(
            root=train_dir,
            zscore=config.zscore, transform=transform
        )
        eval_loader = MultiDirLoader(
            root=eval_dir,
            zscore=False, transform=train_loader.transform
        )
        train_dataset = DiffGraphDataset(loader=train_loader)
        eval_dataset = DiffGraphDataset(loader=eval_loader)
    else:
        raise ValueError("Unknown dataset type: " + config.dataset_type)

    tf.logging.info(
        "Load training dataset from {} successfully: "
        "#examples = {}, observations.shape = {}".format(
            train_dir, len(train_dataset),
            train_dataset[0].observations.shape
        )
    )
    tf.logging.info(
        "Load evaluation dataset from {} successfully: "
        "#examples = {}, observations.shape = {}".format(
            eval_dir, len(eval_dataset),
            eval_dataset[0].observations.shape
        )
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True, drop_last=True,
        collate_fn=batch_and_mask
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=batch_and_mask
    )
    return train_dataloader, eval_dataloader, train_dataset


class AbstractDataset(Dataset):
    def __init__(self, loader):
        super(AbstractDataset, self).__init__()
        self._loader = loader

    @property
    def transform(self):
        return self._loader.transform

    @property
    def dim_observs(self):
        return self._loader.dim_observs

    @property
    def dim_node_attrs(self):
        return self._loader.dim_node_attrs

    @property
    def dim_edge_attrs(self):
        return self._loader.dim_edge_attrs

    @property
    def dim_time_attrs(self):
        return self._loader.dim_time_attrs


class SameGraphDataset(AbstractDataset):
    def __init__(self, loader, win_size=50, stride=50):
        super(SameGraphDataset, self).__init__(loader)

        next_fn = loader.next_example_fn()
        next_fn = sliding_window_wrapper(
            next_fn, win_size=win_size, stride=stride,
            enable=(win_size > 0)
        )
        examples, EOF = [], False
        while not EOF:
            example, EOF = next_fn()
            examples.append(example)

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DiffGraphDataset(AbstractDataset):
    def __init__(self, loader):
        super(DiffGraphDataset, self).__init__(loader)

    def __len__(self):
        return self._loader.num_examples

    def __getitem__(self, idx):
        example, _ = self._loader.access(idx).next_example_fn()()
        return example

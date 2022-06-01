import os
import random
import rasterio
import numpy as np
import itertools
import torch
from random import shuffle
from torch.utils.data import Sampler
import torch.nn.utils.rnn as rnn_util


def load_data(dataPath, isLabel=False):
    """Load the dataset.
    Args:
        dataPath (str) -- Path to either the image or label raster.
        isLabel (binary) -- decide wether the input dataset is label. Default is False.

    Returns:
        loaded data as numpy ndarray.
    """

    if isLabel:

        with rasterio.open(dataPath, "r") as src:

            if src.count != 1:
                raise ValueError("Label must have only 1 band but {} bands were detected.".format(src.count))
            img = src.read(1)

    else:
        img = np.load(dataPath)

    return img


def get_test_pixel_coord(img_cube):
    x_ls = range(img_cube.shape[1])
    y_ls = range(img_cube.shape[2])
    index = list(itertools.product(x_ls, y_ls))

    return index


def make_reproducible(seed=42, cudnn=True):
    """Make all the randomization processes start from a shared seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cudnn:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class CropTypeBatchSampler(Sampler):
    """
    This sampler is designed to divide samples into batches for mini-batch training in a way that samples in each batch
    are closest in sequence length to each other which is helpful as the samples in a batch require the minimum amount of
    zero padding to become equal length.

    Args:
            dataset (Pytorch dataset): list of tuples in the form of [(s1_img, s2_img, label),...,(s1_img, s2_img, label)]
            batch_size (int): Number of samples in a mini-batch training strategy.
            sort_src (str) -- image dataset used for sorting.
            drop_last (bool) -- Decide whether keep or drop the last batch if its length is shorter than batch size.
    Returns:
            list of batches where each batch is a list of sample indices.

    Note 1: Batches are designed so that samples in a batch are closest in sequence length only for the chosen image source.
    Note 2: The last batch might be shorter that the other batch size if drop_last is False.
    Note 3: Separate padding might be required for both sources using 'collate_fn'.
    """

    def __init__(self, dataset, batch_size, sort_src, drop_last=False):
        super(CropTypeBatchSampler, self).__init__(dataset)

        assert sort_src in ["s1", "s2"]
        self.batch_size = batch_size
        self.batches = []
        batch = []
        indices_n_lengths = []

        for i in range(len(dataset)):
            if sort_src == "s1":
                indices_n_lengths.append((i, dataset[i][0].shape[0]))
            else:
                indices_n_lengths.append((i, dataset[i][1].shape[0]))

        shuffle(indices_n_lengths)
        indices_n_lengths.sort(key=lambda x: x[1])

        for i in range(len(indices_n_lengths)):
            sample_idx = indices_n_lengths[i][0]
            batch.append(sample_idx)

            if len(batch) == self.batch_size:
                self.batches.append(batch)
                batch = []

        if len(dataset) % self.batch_size != 0:
            if (len(batch) > 0) and (not drop_last):
                self.batches.append(batch)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for b in self.batches:
            yield b


def collate_var_length(batch):
    batch_size = len(batch)

    s1_max_len = 50
    s2_max_len = 50

    labels = [batch[i][2] for i in range(batch_size)]
    label = torch.stack(labels)

    s1_grids = [batch[i][0] for i in range(batch_size)]
    s2_grids = [batch[i][1] for i in range(batch_size)]

    s1_img = rnn_util.pad_sequence(s1_grids, batch_first=True)
    s2_img = rnn_util.pad_sequence(s2_grids, batch_first=True)

    return s1_img, s2_img, label

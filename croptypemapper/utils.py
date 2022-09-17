import os
import random
import rasterio
import numpy as np
import itertools
import torch
from random import shuffle
from torch.utils.data import Sampler
import torch.nn.utils.rnn as rnn_util
from torch.optim.lr_scheduler import _LRScheduler


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

##########################################################################

def get_test_pixel_coord(img_cube):
    x_ls = range(img_cube.shape[1])
    y_ls = range(img_cube.shape[2])
    index = list(itertools.product(x_ls, y_ls))

    return index

##########################################################################

def collate_var_length(batch):
    
    batch_size = len(batch)

    labels = [batch[i][2] for i in range(batch_size)]
    label = torch.stack(labels)

    s1_grids = [batch[i][0] for i in range(batch_size)]
    s2_grids = [batch[i][1] for i in range(batch_size)]
    
    #s1_lengths = [batch[i][0].shape[0] for i in range(batch_size)]
    #s2_lengths = [batch[i][1].shape[0] for i in range(batch_size)]
    #s1_batch = torch.nn.utils.rnn.pack_padded_sequence(s1_padded, s1_lengths, batch_first=True, enforce_sorted=False)
    #s2_batch = torch.nn.utils.rnn.pack_padded_sequence(s2_padded, s2_lengths, batch_first=True, enforce_sorted=False)

    s1_img = rnn_util.pad_sequence(s1_grids, batch_first=True)
    s2_img = rnn_util.pad_sequence(s2_grids, batch_first=True)

    return s1_img, s2_img, label

############################################################

def cnn_collate_var_length(batch):
    #set_trace()
    batch_size = len(batch)
    
    s1_max_len = config["train_max_sequence_length"][0]
    s2_max_len = config["train_max_sequence_length"][1]
    
    labels = [batch[i][2] for i in range(batch_size)]
    label = torch.stack(labels)
    
    s1_grids = [batch[i][0] for i in range(batch_size)]
    s2_grids = [batch[i][1] for i in range(batch_size)]
    
    s1_grids[0] = nn.ConstantPad1d((0, s1_max_len - s1_grids[0].shape[0]), 0)(torch.transpose(s1_grids[0], 1, 0))
    s1_grids[0] = torch.transpose(s1_grids[0], 1, 0)
    s2_grids[0] = nn.ConstantPad1d((0, s2_max_len - s2_grids[0].shape[0]), 0)(torch.transpose(s2_grids[0], 1, 0))
    s2_grids[0] = torch.transpose(s2_grids[0], 1, 0)

    
    s1_img = rnn_util.pad_sequence(s1_grids, batch_first=True)
    s2_img = rnn_util.pad_sequence(s2_grids, batch_first=True)
    
    return s1_img, s2_img, label 

##########################################################################

def get_optimizer(optimizer, params, lr, momentum):

    optimizer = optimizer.lower()
    if optimizer == 'sgd':
        return torch.optim.SGD(params, lr, momentum=momentum)
    elif optimizer == 'nesterov':
        return torch.optim.SGD(params, lr, momentum=momentum, nesterov=True)
    elif optimizer == 'adam':
        return torch.optim.Adam(params, lr)
    elif optimizer == 'amsgrad':
        return torch.optim.Adam(params, lr, amsgrad=True)
    else:
        raise ValueError("{} currently not supported, please customize your optimizer in compiler.py".format(optimizer))

##########################################################################

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
    
    def __init__(self, dataset, batch_size, sort_src, drop_last=False, shuffleBatches=True):
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
        indices_n_lengths.sort(key = lambda x:x[1])
        
        for i in range(len(indices_n_lengths)):
            sample_idx = indices_n_lengths[i][0]
            batch.append(sample_idx)
            
            if len(batch) == self.batch_size:
                self.batches.append(batch)
                batch = []
        
        if len(dataset) % self.batch_size != 0:       
            if (len(batch) > 0) and (not drop_last):
                self.batches.append(batch)
        
        if shuffleBatches is True:
            random.shuffle(self.batches)
    
    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        for b in self.batches:
            yield b

##########################################################################

class PolynomialLR(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        min_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, min_learning_rate=1e-5, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.min_learning_rate = min_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.min_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.min_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                self.min_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.min_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                         self.min_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


def pickle_dataset(dataset, filePath):
    with open(filePath, "wb") as fp:
        pickle.dump(dataset, fp)

def load_dataset(filePath):
    return pd.read_pickle(filePath)


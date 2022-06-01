import os
from pathlib import Path
import random
import tqdm
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Utils import get_test_pixel_coord


class pixelDataset(Dataset):
    """
    Args:
            root_dir (str): path to the main folder of the dataset, formatted as indicated in the readme
            usage (str): decide whether we are making a "train", "validation" or "test" dataset.
            num_samples (int) -- Number of samples for each crop type.
            sampling_strategy (str) -- If ranked samples are only taken from crop pixels with the lowest number of cloudy days.
                                       Otherwise, a samples can be chosen randomly from the all the available samples for each crop type.
            sources (list of str): Sensors of image acquisition. At the moment two sensors
                                   are used ["Sentinel-1", "Sentinel-2"]
            inference_index (iterable) : Only gets used at prediction time as a mechanism to go through prediction tiles one at a time.
            verbose (bool): Decide to print extra information on-screen.
    """

    def __init__(self, root_dir, usage, num_samples, sampling_strategy="ranked", sources=("Sentinel-1", "Sentinel-2"),
                 inference_index=None, verbose=False):

        self.usage = usage
        self.sources = sources
        self.num_samples = num_samples
        self.sampling_strategy = sampling_strategy

        assert self.usage in ["train", "validation",
                              "test"], "Usage can only be one of 'train', 'validation' and 'test'."
        assert self.sampling_strategy in ["ranked", "random"], "Sampling strategy is invalid."

        if self.usage in ["train", "validation"]:

            s1_dir = Path(root_dir).joinpath(self.sources[0], self.usage, "categories")
            s2_dir = Path(root_dir).joinpath(self.sources[1], self.usage, "categories")
            categories = [name for name in os.listdir(s1_dir) if os.path.isdir(os.path.join(s1_dir, name))]

            s1_samples_ls = []
            s2_samples_ls = []

            for cat in categories:
                s1_src_path = Path(s1_dir).joinpath(cat)
                s1_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s1_src_path) for \
                             f in filenames if f.endswith(".npy")]
                s1_fnames.sort()

                s2_src_path = Path(s2_dir).joinpath(cat)
                s2_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s2_src_path) for \
                             f in filenames if f.endswith(".npy")]
                s2_fnames.sort()
                assert len(s1_fnames) == len(s2_fnames)

                if len(s1_fnames) < self.num_samples:
                    s1_samples_ls.extend(s1_fnames)
                    s2_samples_ls.extend(s2_fnames)
                    print(f"===> Only {len(s1_fnames)} samples are available for {cat} class. All taken.")

                else:
                    if sampling_strategy == "ranked":

                        grid_numbers = [str(f).split("_")[-5] for f in s1_fnames]
                        num_unique_tiles = len(sorted(set(grid_numbers)))
                        min_num_samp_from_each_id = self.num_samples // num_unique_tiles

                        if verbose:
                            print(
                                f"Category: {cat}, Total number of tiles: {num_unique_tiles} Num samples per ite: {min_num_samp_from_each_id}")
                            print("#####")
                        i = 1
                        counter = 0
                        while counter < self.num_samples:
                            for grid in sorted(set(grid_numbers)):
                                s1_samples_in_grid = [str(f) for f in s1_fnames if "_" + grid + "_" in str(f)]
                                s2_samples_in_grid = [str(f) for f in s2_fnames if "_" + grid + "_" in str(f)]
                                assert len(s1_samples_in_grid) == len(s2_samples_in_grid)
                                if len(s1_samples_in_grid) > (i + min_num_samp_from_each_id):
                                    diff = abs(self.num_samples - counter)
                                    if diff >= min_num_samp_from_each_id:
                                        s1_samples = [fn for fn in s1_samples_in_grid if
                                                      int(str(fn).split("_")[-3]) in range(i,
                                                                                           i + min_num_samp_from_each_id)]
                                        s2_samples = [fn for fn in s2_samples_in_grid if
                                                      int(str(fn).split("_")[-3]) in range(i,
                                                                                           i + min_num_samp_from_each_id)]
                                    else:
                                        s1_samples = [fn for fn in s1_samples_in_grid if
                                                      int(str(fn).split("_")[-3]) in range(i, i + diff)]
                                        s2_samples = [fn for fn in s2_samples_in_grid if
                                                      int(str(fn).split("_")[-3]) in range(i, i + diff)]
                                else:
                                    else_diff = len(s1_samples_in_grid) - i
                                    s1_samples = [fn for fn in s1_samples_in_grid if
                                                  int(str(fn).split("_")[-3]) in range(i, i + else_diff)]
                                    s2_samples = [fn for fn in s2_samples_in_grid if
                                                  int(str(fn).split("_")[-3]) in range(i, i + else_diff)]

                                if verbose:
                                    print(f"grid: {grid}, counter: {counter}, i: {i}")
                                    print(f"S1 samples: {s1_samples}")
                                    print("")
                                    print(f"S2 samples: {s2_samples}")
                                    print("-----")
                                s1_samples_ls.extend(s1_samples)
                                s2_samples_ls.extend(s2_samples)
                                counter += len(s1_samples)
                                if counter >= self.num_samples:
                                    break
                        i += min_num_samp_from_each_id

                    else:
                        random_indices = random.sample(range(len(s1_fnames)), self.num_samples)
                        for idx in random_indices:
                            s1_samples_ls.append(s1_fnames[idx])
                            s2_samples_ls.append(s2_fnames[idx])

            self.lbl = []
            self.s1 = []
            self.s2 = []

            assert len(s1_samples_ls) == len(s2_samples_ls)

            for s1_fn, s2_fn in tqdm.tqdm(zip(s1_samples_ls, s2_samples_ls), total=len(s1_samples_ls)):
                s1_lbl = str(s1_fn).split("_")[-1].replace(".npy", "")
                s2_lbl = str(s2_fn).split("_")[-1].replace(".npy", "")
                assert s1_lbl == s2_lbl

                lbl_val = int(s1_lbl)
                self.lbl.append(lbl_val)

                s1_array = np.load(s1_fn)
                self.s1.append(s1_array)

                s2_array = np.load(s2_fn)
                self.s2.append(s2_array)

            print(
                "------{} tuple samples of form (s1, s2, lbl) are loaded from the {} dataset------".format(len(self.s1),
                                                                                                           self.usage))

        if self.usage == "test":

            self.s1 = []
            self.s2 = []
            self.img_coor = []

            s1_dir = Path(root_dir).joinpath(self.sources[0])
            s1_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s1_dir) for f in filenames if
                         f.endswith(".npy")]
            s1_meta_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s1_dir) for f in filenames
                              if f.endswith(".pickle")]
            s1_fnames.sort()
            s1_meta_fnames.sort()

            s2_dir = Path(root_dir).joinpath(self.sources[1])
            s2_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s2_dir) for f in filenames if
                         f.endswith(".npy")]
            s2_meta_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s2_dir) for f in filenames
                              if f.endswith(".pickle")]
            s2_fnames.sort()
            s2_meta_fnames.sort()

            s1_grid_id = str(s1_fnames[inference_index]).split("_")[-1].replace(".npy", "")
            s2_grid_id = str(s2_fnames[inference_index]).split("_")[-1].replace(".npy", "")
            s1_grid_meta = str(s1_meta_fnames[inference_index]).split("_")[-2]
            s2_grid_meta = str(s2_meta_fnames[inference_index]).split("_")[-2]
            assert s1_grid_id == s2_grid_id == s1_grid_meta == s2_grid_meta

            self.tile_id = s1_grid_id
            self.meta = pd.read_pickle(s1_meta_fnames[0])

            s1_array = np.load(s1_fnames[inference_index])
            # s1_array = nn.ConstantPad1d((0, s1_max_len - s1_array.shape[0]), 0)(torch.transpose(s1_array, 1, 0))
            # s1_array = torch.transpose(s1_array, 1, 0)

            s2_array = np.load(s2_fnames[inference_index])
            # s2_array = nn.ConstantPad1d((0, s2_max_len - s2_array.shape[0]), 0)(torch.transpose(s2_array, 1, 0))
            # s2_array = torch.transpose(s2_array, 1, 0)

            assert s1_array.shape[1] == s2_array.shape[1]
            assert s1_array.shape[2] == s2_array.shape[2]

            pixel_indices = get_test_pixel_coord(s1_array)

            for coord in pixel_indices:
                s1_val = s1_array[:, coord[0], coord[1], :]
                self.s1.append(s1_val.copy())

                s2_val = s2_array[:, coord[0], coord[1], :]
                self.s2.append(s2_val.copy())

                self.img_coor.append(coord)

            del s1_array, s2_array
            gc.collect()

    def __getitem__(self, index):

        if self.usage in ["train", "validation"]:
            s1_img = self.s1[index]
            s2_img = self.s2[index]
            label = self.lbl[index]

            # numpy to torch
            # tensor shape: (N x C x T)
            s1_img = torch.from_numpy(s1_img.transpose((1, 0))).float()
            s2_img = torch.from_numpy(s2_img.transpose((1, 0))).float()
            label = torch.from_numpy(np.asarray(label)).long()

            return s1_img, s2_img, label

        if self.usage == "test":
            s1_img = self.s1[index]
            s2_img = self.s2[index]
            coord = self.img_coor[index]

            s1_img = torch.from_numpy(s1_img.transpose((1, 0))).float()
            s2_img = torch.from_numpy(s2_img.transpose((1, 0))).float()

            return s1_img, s2_img, coord

    def __len__(self):
        return len(self.s1)

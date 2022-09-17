import os
from pathlib import Path
import random
import tqdm
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils import get_test_pixel_coord


class pixelDataset(Dataset):
    """
    Args:
            root_dir (str): path to the main folder of the dataset, formatted as indicated in the readme
            usage (str): decide whether we are making a "train", "validation" or "test" dataset.
            region (list of str): a list of names for the geographies of different datasets.
            num_samples (int) -- Number of samples for each crop type.
            sources (list of str): Sensors of image acquisition. At the moment two sensors
                                   are used ["Sentinel-1", "Sentinel-2"]
            inference_index (iterable) : Only gets used at prediction time as a mechanism to go through prediction tiles one at a time.
            verbose (bool): Decide to print extra information on-screen.
    """

    def __init__(self, root_dir, usage, region, num_samples=None, sources=("Sentinel-1", "Sentinel-2"), 
                 inference_dir=None, inference_index=None, verbose=False):

        self.usage = usage
        self.sources = sources
        self.num_samples = num_samples
        self.region = region
        self.inference_dir = inference_dir

        assert self.usage in ["train", "validation", "test"], "Usage can only be one of 'train', 'validation' and 'test'."

        if self.usage in ["train", "validation"]:

            assert num_samples is not None

            s1_samples_ls = []
            s2_samples_ls = []

            for region in self.region:
                print(f"Running {region}")
                s1_dir = Path(root_dir).joinpath(region, self.sources[0], self.usage, "categories")
                s2_dir = Path(root_dir).joinpath(region, self.sources[1], self.usage, "categories")
                categories = [name for name in os.listdir(s1_dir) if os.path.isdir(os.path.join(s1_dir, name))]

                for cat in categories:
                    print(f"...processing {cat}")
                    s1_src_path = Path(s1_dir).joinpath(cat)
                    print(f"...... {s1_src_path}")
                    s1_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s1_src_path) for \
                                f in filenames if f.endswith(".npy")]
                    s1_fnames.sort()

                    s2_src_path = Path(s2_dir).joinpath(cat)
                    print(f"...... {s2_src_path}")
                    
                    s2_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s2_src_path) for \
                                f in filenames if f.endswith(".npy")]
                    s2_fnames.sort()
                    assert len(s1_fnames) == len(s2_fnames)

                    if len(s1_fnames) < self.num_samples:
                        s1_samples_ls.extend(s1_fnames)
                        s2_samples_ls.extend(s2_fnames)
                        print(f"===> Only {len(s1_fnames)} samples are available for {cat} class. All taken.")

                    else:
                        random_indices = random.sample(range(len(s1_fnames)), self.num_samples)
                        print(f"===> {self.num_samples} samples are taken for {cat} class.")
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

                # or if shape[0] < shape[1]
                s1_array = np.load(s1_fn)
                s2_array = np.load(s2_fn)

                if s1_array.shape[0] < s1_array.shape[1]:
                    s1_array = s1_array.transpose((1, 0))
                    s2_array = s2_array.transpose((1, 0))
                
                self.s1.append(s1_array)
                self.s2.append(s2_array)

            print(
                "------{} tuple samples of form (s1, s2, lbl) are loaded from the {} dataset------".format(len(self.s1),
                                                                                                           self.usage))

        if self.usage == "test":

            self.s1 = []
            self.s2 = []
            self.img_coor = []

            s1_dir = Path(root_dir).joinpath(self.inference_dir, self.sources[0])
            s1_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s1_dir) for f in filenames if
                         f.endswith(".npy")]
            s1_meta_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s1_dir) for f in filenames
                              if f.endswith(".pickle")]
            s1_fnames.sort()
            s1_meta_fnames.sort()

            s2_dir = Path(root_dir).joinpath(self.inference_dir, self.sources[1])
            s2_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s2_dir) for f in filenames if
                         f.endswith(".npy") if "source" in f]
            s2_meta_fnames = [Path(dirpath) / f for (dirpath, dirnames, filenames) in os.walk(s2_dir) for f in filenames
                              if f.endswith(".pickle")]
            s2_fnames.sort()
            s2_meta_fnames.sort()

            s1_grid_id = str(s1_fnames[inference_index]).split("_")[-1].replace(".npy", "")
            s2_grid_id = str(s2_fnames[inference_index]).split("_")[-1].replace(".npy", "")
            s1_grid_meta_id = str(s1_meta_fnames[inference_index]).split("_")[-2]
            s2_grid_meta_id = str(s2_meta_fnames[inference_index]).split("_")[-2]
            assert s1_grid_id == s2_grid_id == s1_grid_meta_id == s2_grid_meta_id

            self.tile_id = s1_grid_id
            self.meta = pd.read_pickle(s1_meta_fnames[0])

            s1_array = np.load(s1_fnames[inference_index])
            s1_array = s1_array * 1e-7
            #s1_array_padded = np.zeros((s1_array.shape[0], s1_array.shape[1], s1_array.shape[2], s1_sequence_length), dtype=float)
            #s1_array_padded[:s1_array.shape[0], :s1_array.shape[1], :s1_array.shape[2],:s1_array.shape[3]] = s1_array

            s2_array = np.load(s2_fnames[inference_index])
            s2_array = s2_array * 1e-7
            #s2_array_padded = np.zeros((s2_array.shape[0], s2_array.shape[1], s2_array.shape[2], s2_sequence_length), dtype=float)
            #s2_array_padded[:s2_array.shape[0], :s2_array.shape[1], :s2_array.shape[2],:s2_array.shape[3]] = s2_array

            assert s1_array.shape[1] == s2_array.shape[1]
            assert s1_array.shape[2] == s2_array.shape[2]

            pixel_indices = get_test_pixel_coord(s1_array)
            #pixel_indices = get_test_pixel_coord(s1_array_padded)

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
            # If using Rustowicz dataset un-comment the next two lines and comment out
            # the other two. Otherwise no change
            # s1_img = torch.from_numpy(s1_img.transpose((1, 0))).float()
            # s2_img = torch.from_numpy(s2_img.transpose((1, 0))).float()
            s1_img = torch.from_numpy(s1_img).float()
            s2_img = torch.from_numpy(s2_img).float()
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

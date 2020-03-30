import os
import re
import sys
import glob
import subprocess as sub

import torch
import pandas as pd
import pybrite as pb
from tqdm import tqdm
# from torch_geometric.data import Dataset
from .dataset import Dataset

from .utils import from_networkx


class Brite(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, type_db=None, debug=False):#, version="v1.0", id_folder="", secrets_path=None):
        self.type_db = type_db
        self.debug = debug
        self._size = None
        self._raw_file_names = None
        self._processed_file_names = None
        super(Brite, self).__init__(root, transform, pre_transform)

    @property
    def raw_dir(self):
        if not self.type_db:
            return os.path.join(self.root, "raw")
        return os.path.join(self.root, self.type_db)

    @property
    def processed_dir(self):
        if not self.type_db:
            return os.path.join(self.root, "processed")
        return os.path.join(self.root, self.type_db + '_processed')

    @property
    def raw_file_names(self):
        if self._raw_file_names is None:
            print("Call raw names")
            if self.debug:
                print("Getting raw data from " + self.raw_dir)
            self._raw_file_names = [p.split("/")[-1] for p in glob.glob(os.path.join(self.raw_dir, "*.gpickle"))]
            if self.debug:
                print("Done {}".format(len(self._raw_file_names)))
        return self._raw_file_names

    @property
    def processed_file_names(self):
        if self._processed_file_names is None:
            print("Call processed names")
            if self.debug:
                print("Getting processed data from " + self.processed_dir)
            self._processed_file_names = [p.split("/")[-1] for p in glob.glob(os.path.join(self.processed_dir, "data_*.pt"))]
            if self.debug:
                print("Done {}".format(len(self._processed_file_names)))
        return self._processed_file_names

    @property
    def info(self):
        path = os.path.join(self.raw_dir, "info.dat")
        if os.path.isfile(path):
            dfinfo = pd.read_csv(path)
            n_interval = tuple(dfinfo.loc["n", ["min", "max"]].values)
            m_interval = tuple(dfinfo.loc["m", ["min", "max"]].values)
        else:
            n_interval = ()
            m_interval = ()
        return n_interval, m_interval

    def download(self):
        raise ValueError("There are not any raw data in the {}.".format(self.raw_dir))

    def process(self):
        for i, raw_path in enumerate(self.raw_paths):
            parse_raw_data = pb.read_from_files([raw_path], bidim_solution=False)
            input_nx, target_nx = parse_raw_data[0][0], parse_raw_data[1][0]
            data = from_networkx(input_nx, target_nx)
            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))

    def len(self):
        if self._size is None:
            self._size = len(self.processed_file_names)
        else:
            print("Skip")
        return self._size

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

# def files_exist(files):
#     return all([os.path.exists(f) for f in files]) if len(files) else False

# def __repr__(obj):
#     if obj is None:
#         return 'None'
#     return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

# class Brite(Dataset):
#     def __init__(self, root, size, n_interval, m_interval=(2, 2), transform=None, pre_transform=None, pre_filter=None):
#         super(Dataset, self).__init__()

#         if isinstance(root, str):
#             root = os.path.expanduser(os.path.normpath(root))

#         self.root = root
#         self.transform = transform
#         self.pre_transform = pre_transform
#         self.pre_filter = pre_filter
#         self.__indices__ = None

#         self.size = size
#         self.n_interval = n_interval
#         self.m_interval = m_interval

#         if 'download' in self.__class__.__dict__.keys():
#             self._download()

#         if 'process' in self.__class__.__dict__.keys():
#             self._process()

#     @property
#     def raw_file_names(self):
#         path = Path(self.raw_dir)
#         return [p.stem + ".gpickle" for p in path.glob("*.gpickle")]

#     @property
#     def processed_file_names(self):
#         path = Path(self.processed_dir)
#         return [p.stem + ".pt" for p in path.glob("*.pt") if re.match(r"data_\d+", p.stem)]

#     @property
#     def info(self):
#         path = os.path.join(self.raw_dir, "info.dat")
#         if os.path.isfile(path):
#             dfinfo = pd.read_csv(path)
#             n_interval = tuple(dfinfo.loc["n", ["min", "max"]].values)
#             m_interval = tuple(dfinfo.loc["m", ["min", "max"]].values)
#         else:
#             n_interval = ()
#             m_interval = ()
#         return n_interval, m_interval

#     def _download(self):
#         if files_exist(self.raw_paths):  # pragma: no cover
#             n_interval, m_interval = self.info
#             if self.size == len(self.raw_file_names) and self.n_interval == n_interval and self.m_interval == m_interval:
#                 return
#             else:
#                 for file in self.raw_paths: os.remove(file)
#                 for file in self.processed_paths: os.remove(file)

#         print('Downloading...')
#         if not os.path.exists(self.raw_dir):
#             os.makedirs(self.raw_dir)
#         self.download()
#         print('Done!')

#     def _process(self):
#         f = os.path.join(self.processed_dir, 'pre_transform.pt')
#         if os.path.exists(f) and torch.load(f) != __repr__(self.pre_transform):
#             warnings.warn(
#                 'The `pre_transform` argument differs from the one used in '
#                 'the pre-processed version of this dataset. If you really '
#                 'want to make use of another pre-processing technique, make '
#                 'sure to delete `{}` first.'.format(self.processed_dir))
#         f = os.path.join(self.processed_dir, 'pre_filter.pt')
#         if os.path.exists(f) and torch.load(f) != __repr__(self.pre_filter):
#             warnings.warn(
#                 'The `pre_filter` argument differs from the one used in the '
#                 'pre-processed version of this dataset. If you really want to '
#                 'make use of another pre-fitering technique, make sure to '
#                 'delete `{}` first.'.format(self.processed_dir))

#         if len(self.processed_file_names) == len(self.raw_file_names):  # pragma: no cover
#             return

#         print('Processing...')

#         if not os.path.exists(self.processed_dir):
#             os.makedirs(self.processed_dir)
#         self.process()

#         path = os.path.join(self.processed_dir, 'pre_transform.pt')
#         torch.save(__repr__(self.pre_transform), path)
#         path = os.path.join(self.processed_dir, 'pre_filter.pt')
#         torch.save(__repr__(self.pre_filter), path)

#         print('Done!')

#     def download(self):
#         pb.create_static_dataset(self.raw_dir, self.size, self.n_interval, self.m_interval)

#     def process(self):
#         for i, raw_path in enumerate(self.raw_paths):
#             parse_raw_data = pb.read_from_files([raw_path], bidim_solution=False)
#             input_nx, target_nx = parse_raw_data[0][0], parse_raw_data[1][0]
#             data = from_networkx(input_nx, target_nx)
#             torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))

#     def len(self):
#         return len(self.processed_file_names)

#     def get(self, idx):
#         data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
#         return data

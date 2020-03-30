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
            self._raw_file_names = [p.split("/")[-1] for p in glob.glob(os.path.join(self.raw_dir, "*.gpickle"))]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        if self._processed_file_names is None:
            print("Call processed names")
            self._processed_file_names = [p.split("/")[-1] for p in glob.glob(os.path.join(self.processed_dir, "data_*.pt"))]
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
        return self._size

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

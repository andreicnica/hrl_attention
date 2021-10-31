import os
import torch
import numpy as np
import shutil
import itertools
import glob
import re
from typing import Any

FOLDER_CHECKPOINT = "checkpoints/"
FILE_CHECKPOINT_BEST = "training_data_best.pt"
FILE_CHECKPOINT = "training_data_{}.pt"


def get_training_data_path(out_dir: str, best: bool = False, index: int = None) -> str:
    if best:
        return os.path.join(out_dir, FILE_CHECKPOINT_BEST)

    if index is not None:
        fld = os.path.join(out_dir, FOLDER_CHECKPOINT)
        if not os.path.isdir(fld):
            os.mkdir(fld)
        return os.path.join(fld, FILE_CHECKPOINT.format(index))

    return os.path.join(out_dir, FILE_CHECKPOINT.format(""))


def get_last_training_path_idx(out_dir: str) -> int:
    if os.path.exists(out_dir):
        path = os.path.join(out_dir, "training_data_*.pt")

        max_index = 0
        for path in glob.glob(path):
            try:
                max_index = max(max_index,
                                int(re.findall("training_data_([1-9]\d*|0).pt", path)[0]))
            except:
                 pass

        return max_index
    return 0


class SaveData:
    def __init__(self, out_dir: str, save_best: bool = True, save_all: bool = False):
        self.out_dir = out_dir
        self.save_best = save_best
        self.save_all = save_all

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.best_reward = -np.inf

        start_idx = get_last_training_path_idx(out_dir)
        self.index = itertools.count(start=start_idx, step=1)

    def load_training_data(self, out_dir: str = None, best: bool = False, index: int = None):
        """ If best is set to false, the last training model is loaded """
        out_dir = out_dir if out_dir is not None else self.out_dir
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        training_data = None
        if best:
            path = get_training_data_path(out_dir, best=best)
            if os.path.isfile(path):
                training_data = torch.load(path, map_location=map_location)

        if training_data is None:
            path = get_training_data_path(out_dir, best=False, index=index)
            try:
                training_data = torch.load(path, map_location=map_location)
            except OSError:
                training_data = dict({"model": None, "agent": {}})

        if "eprew" in training_data:
            self.best_reward = training_data["eprew"]

        return training_data.pop("model"), training_data.pop("agent"), training_data

    def save_training_data(self, model: Any, agent: Any, eprew: float,
                           other: dict = None, out_dir: str = None):
        out_dir = out_dir if out_dir is not None else self.out_dir

        trainig_data = dict()
        trainig_data["model"] = model
        trainig_data["agent"] = agent
        trainig_data["eprew"] = eprew

        if other is not None:
            trainig_data.update(other)

        # Save standard
        path = get_training_data_path(out_dir)
        torch.save(trainig_data, path)

        if eprew > self.best_reward:
            self.best_reward = eprew
            best_path = get_training_data_path(out_dir, best=True)
            shutil.copyfile(path, best_path)

        if self.save_all:
            index_path = get_training_data_path(out_dir, index=next(self.index))
            shutil.copyfile(path, index_path)

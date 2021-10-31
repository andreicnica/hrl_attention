import csv
import json
import logging
import sys
from os.path import join
from typing import TextIO, Tuple, Any
import time
import numpy as np
from collections import OrderedDict
from typing import List
from argparse import Namespace
from collections import namedtuple, deque

from utils import utils


LogCfg = namedtuple(
    "LogCfg", ["key_name", "fixed_window", "processing", "abbreviation", "print_format", "print",
               "plot"]
)
"""
    key_name: Key to uniquely identify data (not print log) 
    fixed_window: True, If statistics should be calculated on a fixed number of values defined by 
        stats_window_size from cfg.main. If so, than we will keep all previous 
        (stats_window_size) values and calculate processing on it.
        If not. list will be cleared after each log.
    processing: How to summarize list (mean, average, etc.)  See LOG_TYPES for all possible types.
    abbreviation: Used for print
    print_format: e.g. for print("string_{:.2f}) -> :.2f
    print: If it should be printed
    plot: If it should be plotted
"""

LOG_TYPES = {
    "a": np.mean,  # average list
    "s": np.std,  # standard deviation list
    "m": np.min,  # min from list
    "M": np.max,  # max from list
    "l": lambda x: x[-1],  # last element in list
}


class DataList:
    def __init__(self, fixed_window, values):
        self.fixed_window = fixed_window
        self.values = values


FILE_LOG = "log.txt"
FILE_CSV = "log.csv"
FILE_VOCAB = "vocab.json"
FILE_STATUS = "status.json"


DEFAULT_TRAIN_LOG = [
    LogCfg("update", False, "l", "U", "", True, True),
    LogCfg("num_frames", False, "l", "F", ":06", True, True),
    LogCfg("num_frames_no_off", False, "l", "F0", ":06", False, True),
    LogCfg("fps", False, "a", "FPS", ":04.0f", True, False),
    LogCfg("duration", False, "l", "D", "", True, False),
]


def extra_log_fields(header: list, log_keys: list) ->list:
    unusable_fields = ['return_per_episode', 'reshaped_return_per_episode',
                       'num_frames_per_episode', 'num_frames']
    extra_fields = []
    for field in log_keys:
        if field not in header and field not in unusable_fields:
            extra_fields.append(field)

    return extra_fields


def get_logger(log_dir: str) -> logging.Logger:
    if log_dir is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )
    else:
        path = join(log_dir, FILE_LOG)
        utils.create_folders_if_necessary(path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(filename=path),
                logging.StreamHandler(sys.stdout)
            ]
        )

    return logging.getLogger()


def get_csv_writer(log_dir: str, csv_file_name: str) -> Tuple[TextIO, Any]:
    csv_path = join(log_dir, csv_file_name)
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


def load_status(log_dir: str) -> dict:
    path = join(log_dir, FILE_STATUS)
    with open(path) as file:
        return json.load(file)


class Header:
    def __init__(self, header_config: List[LogCfg], window_size: int = 1, max_buffer: int = 9999):
        self.window_size = window_size
        self.header = header = OrderedDict()
        self.data = data = dict()
        self._max_buffer = max_buffer

        for ix, cfg in enumerate(header_config):
            key_name = f"{cfg.key_name}_{cfg.processing}"
            key = cfg.key_name
            header[key_name] = Namespace()
            header[key_name].key = key
            header[key_name].process = LOG_TYPES[cfg.processing]
            header[key_name].abbr = cfg.abbreviation
            header[key_name].print_str = cfg.abbreviation + " {" + cfg.print_format + "}"
            header[key_name].print = cfg.print
            header[key_name].fixed_window = cfg.fixed_window
            header[key_name].plot = cfg.plot

            # data window size is configured according to first cfg
            if key not in self.data:
                if cfg.fixed_window:
                    # TODO List growing too big
                    data[key] = DataList(True, [0])
                else:
                    data[key] = DataList(False, [])

    def __getitem__(self, item):
        return self.header[item]

    def reset(self):
        window_size = self.window_size

        for k, v in self.data.items():
            if v.fixed_window:
                v.values = v.values[-window_size:]
            else:
                v.values = []

    def add_data(self, data: dict):
        local_data = self.data
        for k, v in data.items():
            local_data[k].values += v  # Must add only list to logs!
            if len(local_data[k].values) > self._max_buffer:
                local_data[k].values = local_data[k].values[-self._max_buffer:]

    def get_csv_header(self) -> List[str]:
        return self.header.keys()

    def get_values(self) -> List[Any]:
        vs = []
        data = self.data
        win_size = self.window_size
        for kkk, v in self.header.items():
            k = v.key
            if len(data[k].values) > 0:
                if data[k].fixed_window:
                    vs.append(v.process(data[k].values[-win_size:]))
                else:
                    vs.append(v.process(data[k].values))
            else:
                vs.append(None)
        return vs

    def get_win_value(self, key, default=None) -> Any:
        data = self.data
        if len(data[key].values) > 0:
            return np.mean(data[key].values[-self.window_size:])
        else:
            return default

    def get_log(self) -> Tuple[List[str], str, dict]:
        data = self.data
        win_size = self.window_size

        csv_row = []
        print_row = []
        plot_values = dict()

        for kkk, v in self.header.items():
            k = v.key
            if len(data[k].values) > 0:
                if data[k].fixed_window:
                    rv = v.process(data[k].values[-win_size:])
                else:
                    rv = v.process(data[k].values)
            else:
                rv = None

            csv_row.append(rv)
            if v.print:
                print_row.append(v.print_str.format(rv))

            if v.plot:
                plot_values[kkk] = rv

        return csv_row, " | ".join(print_row), plot_values


class DummyCSV:
    def __init__(self):
        pass

    def flush(self):
        pass


class DummyCSVWrite:
    def __init__(self):
        pass

    def writerow(self):
        pass


class LoggerMaster:
    def __init__(self, log_dir: str, plot: bool = False, plot_x: str = None,
                 stats_window_size: int = 1,
                 logger=None, plotter=None, csv_file_name: str = FILE_CSV,
                 plot_project: str = "project-0", experiment: str = "default",
                 cfg: dict = None):
        self.log_dir = log_dir
        self.plot_x = plot_x
        self.window_size = stats_window_size

        self.logger = get_logger(log_dir) if logger is None else logger
        if log_dir is not None:
            self.csv_file, self.csv_writer = get_csv_writer(log_dir, csv_file_name)
        else:
            self.csv_file, self.csv_writer = DummyCSV(), DummyCSVWrite()

        if plot:
            import wandb
            import os

            if os.path.isfile(".wandb_key"):
                with open(".wandb_key") as f:
                    WANDB_API_KEY = f.readline().strip()

                os.environ['WANDB_API_KEY'] = WANDB_API_KEY

            wandb.init(project=plot_project, name=f"{experiment}")
            if cfg is not None:
                wandb.config.update(cfg)
            self.plotter = wandb
        else:
            self.plotter = plotter

        self.total_start_time = time.time()
        self.update_start_time = time.time()

        self.header = None  # type: Header

    def get_plotter(self):
        return self.plotter

    def set_header(self, update, header_config: List[LogCfg]):
        header_config = DEFAULT_TRAIN_LOG + header_config
        self.header = Header(header_config, window_size=self.window_size)

        if update <= 1:
            self.csv_writer.writerow(self.header.get_csv_header())

        self.csv_file.flush()

    def toggle_print(self, toggle_print: List[str]):
        for x in toggle_print:
            if x in self.header.header.keys():
                self.header[x].print = not self.header[x].print

    def toggle_plot(self, toggle_plot: List[str]):
        for x in toggle_plot:
            if x in self.header.header.keys():
                self.header[x].plot = not self.header[x].plot

    def get_logger(self):
        return self.logger

    def register_log(self, update: int, update_num_frames: int, logs: dict):
        update_start_time = time.time()

        num_frames = logs["num_frames"][0]
        logs["update"] = [update]
        logs["fps"] = [update_num_frames / (update_start_time - self.update_start_time)]
        logs["duration"] = [int(time.time() - self.total_start_time)]

        self.update_start_time = update_start_time

        self.header.add_data(logs)

    def log_logs(self):

        csv_row, print_row, plot_values = self.header.get_log()

        self.csv_writer.writerow(csv_row)
        self.csv_file.flush()
        self.logger.info(print_row)

        self.header.reset()

        if self.plotter:
            self.plotter.log(plot_values)

from argparse import Namespace
from typing import List
import os
from liftoff import OptionParser, dict_to_namespace
import yaml


def add_to_cfg(cfg: Namespace, subgroups: List[str], new_arg: str, new_arg_value) -> None:
    for arg in subgroups:
        if hasattr(cfg, arg):
            setattr(getattr(cfg, arg), new_arg, new_arg_value)


def create_folders_if_necessary(path: str):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def seed(seed):
    import random
    import numpy
    import torch
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_opts(check_out_dir: bool = True):
    """ This should be called by all scripts prepared by liftoff.

        python script.py results/something/cfg.yaml

        in your script.py

          if __name__ == "__main__":
              from liftoff import parse_opts()
              main(parse_opts())
    """

    opt_parser = OptionParser("liftoff", ["config_path", "session_id", "results_path"])
    opts = opt_parser.parse_args()

    if opts.results_path != "./results":
        change_out_dir = opts.results_path
    else:
        change_out_dir = None

    config_path = opts.config_path
    with open(opts.config_path) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
    opts = dict_to_namespace(config_data)

    if not hasattr(opts, "out_dir"):
        opts.out_dir = f"results/experiment_{os.path.dirname(config_path)}"
        opts.run_id = 1
    if check_out_dir and not os.path.isdir(opts.out_dir):  # pylint: disable=no-member
        os.mkdir(opts.out_dir)
        print(f"New out_dir created: {opts.out_dir}")
    elif change_out_dir is not None:
        opts.out_dir = change_out_dir
    else:
        print(f"Existing out_dir: {opts.out_dir}")

    return opts


def flatten_cfg(cfg: Namespace):
    lst = []
    for key, value in cfg.__dict__.items():
        if isinstance(value, Namespace):
            for key2, value2 in flatten_cfg(value):
                lst.append((f"{key}.{key2}", value2))
        else:
            lst.append((key, value))
    return lst

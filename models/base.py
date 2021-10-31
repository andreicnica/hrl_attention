# Credits (inspired & adapted from) @ https://github.com/lcswillems/torch-rl

import gym
from argparse import Namespace
import torch

from utils.dictlist import DictList


class ModelBase:
    # Not recurrent by default
    recurrent = False

    def __init__(self, cfg: Namespace, obs_space: dict, action_space: gym.spaces):
        self.recurrent = cfg.recurrent
        self.use_text = cfg.use_text

    def forward(self, obs: DictList, memory: torch.Tensor = None, mask: torch.Tensor = None) \
            -> dict:
        """
            :param obs
            :param memory
            :param mask - 0 if new episode - 1 if same episode

            Must return dict with at least the following tensors  (actions, act_log_probs, values).
            -> tensors shape == (len(obs)) . Results for each observations.
        """
        # if not recurrent
        # return dict({"actions": actions, "act_log_probs": act_log_probs, "values": values})
        # elif recurrent - return memory as well
        # {"actions": actions, "act_log_probs": act_log_probs, "values": values, "memory": memory})
        raise NotImplementedError

    @property
    def memory_size(self):
        return None

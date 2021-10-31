import re
import json
import os
import gym
import numpy as np
import torch

from utils.dictlist import DictList
from utils.utils import create_folders_if_necessary
from utils.logging_utils import FILE_VOCAB


def process_simple(env_id, obs_space, out_dir, cfg):
    normalize = getattr(cfg, "normalize", True)
    max_image_value = getattr(cfg, "max_image_value", 10.)
    permute = getattr(cfg, "permute", False)

    # Check if it is a MiniGrid environment
    if re.match("MiniGrid-.*", env_id):
        obs_space = {"image": obs_space.spaces['image'].shape}

        def pre_process_obss(obss, device=None):
            return DictList({
                "image": pre_process_images([obs["image"] for obs in obss], device=device,
                                            max_image_value=max_image_value, normalize=normalize,
                                            permute=permute),
            })

    # Check if the obs_space is of type Box([X, Y, 3])
    elif isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3\
            and obs_space.shape[2] == 3:
        obs_space = {"image": obs_space.shape}

        def pre_process_obss(obss, device=None):
            return DictList({
                "image": pre_process_images(obss, device=device, max_image_value=max_image_value,
                                            normalize=normalize)
            })

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, pre_process_obss


def process_fetch(env_id, obs_space, out_dir, cfg, obs="obs"):
    normalize = getattr(cfg, "normalize", False)
    max_image_value = getattr(cfg, "max_image_value", 1.)

    def pre_process_obss(obss, device=None, ret_obs=obs):
        # TODO should transpose when generating observation

        if ret_obs is None:
            obss = torch.stack(obss).float().to(device)  # type: torch.Tensor
        else:
            obss = torch.stack([torch.from_numpy(x[ret_obs]) for x in obss]).float().to(device)  # type: torch.Tensor

        if normalize:
            obss.div_(max_image_value)
        return obss

    if obs is not None:
        obs_space = obs_space[obs]

    return obs_space, pre_process_obss


def process_simple_with_intents(env_id, obs_space, out_dir, cfg):
    normalize = getattr(cfg, "normalize", True)
    max_image_value = getattr(cfg, "max_image_value", 10.)
    permute = getattr(cfg, "permute", False)

    # Check if it is a MiniGrid environment
    obs_space = {"image": obs_space.spaces['image'].shape}

    def pre_process_obss(obss, device=None):
        obs = dict()

        for k in obss[0].keys():
            if k == "image":
                obs["image"] = pre_process_images([obs["image"] for obs in obss], device=device,
                                                  max_image_value=max_image_value,
                                                  normalize=normalize, permute=permute)
            elif k != "mission":
                obs[k] = pre_process_vec([obs[k] for obs in obss], device=device)

        return DictList(obs)

    return obs_space, pre_process_obss


def pre_process_vec(vec, device=None):
    vec = np.array(vec)
    vec = torch.tensor(vec, device=device)  # type: torch.Tensor
    return vec


def pre_process_images(images, device=None, max_image_value=15., normalize=True, permute=False):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    images = torch.tensor(images, device=device, dtype=torch.float)  # type: torch.Tensor
    if normalize:
        images.div_(max_image_value)
    if permute:
        images = images.permute(0, 3, 1, 2)
    return images


class Vocabulary:
    """
        Copyrights https://github.com/lcswillems/torch-rl

        A mapping from tokens to ids with a capacity of `max_size` words.
        It can be saved in a `vocab.json` file.
    """

    def __init__(self, model_dir, max_size):
        self.path = os.path.join(model_dir, FILE_VOCAB)
        self.max_size = max_size
        self.vocab = {}
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self):
        create_folders_if_necessary(self.path)
        json.dump(self.vocab, open(self.path, "w"))

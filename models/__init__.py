from . import oc_simple
from . import aff_multistep
from . import ac_interest_cnn

import torch
from argparse import Namespace
import gym


MODELS = {
    "AcInterestCNN": ac_interest_cnn.AcInterestCNN,
    "OcSimple": oc_simple.OcSimple,
    "AffMultiStepNet": aff_multistep.AffMultiStepNet,
}


def get_model(cfg: Namespace, obs_space: dict, action_space: gym.spaces, **kwargs) -> \
        torch.nn.Module:
    assert hasattr(cfg, "name") and cfg.name in MODELS,\
        "Please provide a valid model name."
    return MODELS[cfg.name](cfg, obs_space, action_space, **kwargs)

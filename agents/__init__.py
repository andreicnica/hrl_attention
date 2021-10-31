from . import base
from . import ppo
from . import oc
from . import base_affordance
from . import multistep_affordance
from . import ppo_smdp
from . import ppo_ir

import torch
from argparse import Namespace
from typing import List
import gym


AGENTS = {
    "BaseAffordances": base_affordance.BaseAffordances,
    "PPO": ppo.PPO,
    "OC": oc.OptionCritic,
    "MultistepAffordances": multistep_affordance.MultistepAffordances,
    "PPOSmdp": ppo_smdp.PPOSmdp,
    "PPOir": ppo_ir.PPOir,
}


def get_agent(cfg: Namespace, envs: List[gym.Env], model: torch.nn.Module,
              **kwargs) -> base.AgentBase:
    assert hasattr(cfg, "name") and cfg.name in AGENTS,\
        "Please provide a valid Agent name."
    return AGENTS[cfg.name](cfg, envs, model, **kwargs)

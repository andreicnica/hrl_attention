# Credits (inspired & adapted from) @ https://github.com/lcswillems/torch-rl

import torch
from argparse import Namespace
from typing import List, Any, Tuple
import gym
import os
import copy

from utils.dictlist import DictList
from utils.logging_utils import LogCfg
from utils.vec_env import ParallelEnv


BASE_LOGS = [
    LogCfg("return_per_episode", True, "a", "rR:u,", ":.2f", True, True),
    LogCfg("return_per_episode", True, "s", ",s", ":.2f", False, False),
    LogCfg("return_per_episode", True, "m", ",m", ":.2f", False, False),
    LogCfg("return_per_episode", True, "M", ",M", ":.2f", False, False),
    LogCfg("num_frames_per_episode", True, "a", "F:u,", ":.2f", True, False),
    LogCfg("num_frames_per_episode", True, "s", ",s", ":.2f", False, False),
    LogCfg("num_frames_per_episode", True, "m", ",m", ":.2f", False, False),
    LogCfg("num_frames_per_episode", True, "M", ",M", ":.2f", False, False),
]


class AgentBase:
    """ The base class for the RL algorithms. """

    def __init__(
            self, cfg: Namespace, envs: List[gym.Env], model: torch.nn.Module,
            preprocess_obss: Any, eval_envs: List[Any], eval_episodes: int = 0,
            parallel_env_class=ParallelEnv
    ):
        """
        :param cfg: Namespace - Agent config
        :param envs: list[gym_environments] - a list of environments that will be run in parallel
        :param model: torch.Module -
            Agent model - processes observations and returns (action, action_log_prob, value,
            memory)
        :param preprocess_obss : function -
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        :param eval_envs: list[gym_environments]
            - a list of evaluation environments that will be run in parallel
        :param eval_episodes: int
            - Number of episodes that should be evaluated at each "evaluate" call

        cfg Should contain the following (at least for the base):
            :cfg num_frames_per_proc: int -
                the number of frames collected by every process for an update
            :cfg discount: float - the discount for future rewards
            :cfg gae_lambda: float - the lambda coefficient in the GAE formula
            :cfg recurrence: int - the number of steps the gradient is propagated back in time
        """

        # -- Store parameters
        self.cfg = cfg
        # Can load env class directly
        self.env = parallel_env_class(envs) if isinstance(envs, list) else envs
        self.model = model
        self.preprocess_obss = preprocess_obss
        self.eval_envs = eval_envs
        self.eval_episodes = eval_episodes

        # -- No evaluator by default
        self.has_evaluator = len(eval_envs) > 0 and eval_episodes > 0

        # -- Store helpers values
        self.recurrent = model.recurrent
        self.out_dir = cfg.out_dir
        self.device = device = cfg.device
        self.discount = cfg.discount
        self.gae_lambda = cfg.gae_lambda
        self.num_procs = num_procs = self.env.num_procs

        if self.recurrent or hasattr(cfg, "recurrence"):
            self.recurrence = cfg.recurrence
            assert (self.recurrence > 1) == self.recurrent, "Rec>1 but model is not recurrent"

        # -- Initialize log values (Should append if extending agent
        self._logs_train_config = copy.deepcopy(BASE_LOGS)
        self._logs_eval_config = []

        self.log_done_counter = 0
        self.log_return = [0] * num_procs
        self.log_num_frames = [0] * num_procs

        # -- Run useful methods
        self.model.train()

        # -- Init evaluation environments and storage data
        if self.has_evaluator:
            self.eval_envs = parallel_env_class(eval_envs) if isinstance(eval_envs, list) \
                else eval_envs

        # -- Configure agent for saving training data (if save_experience is set)
        # This should be configured as the number of environments to save exp from
        self.save_experience = save_experience = getattr(cfg, "save_experience", 0)

        if save_experience:
            self.experience_dir = f"{self.out_dir}/exp"
            os.mkdir(self.experience_dir)
            assert hasattr(cfg, "max_image_value"), "Need norm value in order to save obs astype " \
                                                    "uint8"
            self.max_image_value = cfg.max_image_value

    def update_parameters(self) -> dict:
        """  [REPLACE] Implement agent training  """

        # -- Collect experiences

        # -- Train Loop
        # ...
        # return logs
        raise NotImplementedError

    def get_checkpoint(self) -> dict:
        """ [REPLACE]
            Return necessary variables for restarting training from checkpoint e.g optimizer
            Same data will be used for loading using load_agent_data
        """
        raise NotImplementedError

    def load_checkpoint(self, agent_data: dict):
        """  [REPLACE] Load agent checkpoint data. e.g. optimizer """
        raise NotImplementedError

    def evaluate(self, eval_key=None) -> dict:
        """ [REPLACE]
            Evaluate agent and return logs. (no evaluater by default)
            Must change self.has_evaluator
            Logs for evaluation should be appended to self._logs_eval_config List
        """
        raise NotImplementedError

    def _collect_experiences(self) -> Tuple[DictList, dict]:
        """ Collects rollouts."""
        # Return experience and logs
        raise NotImplementedError

    def _collect_experiences_step(self, frame_id: int, obs: List[DictList], reward: List[float],
                                  done: List[bool], info: List[dict], prev_obs: DictList,
                                  model_result: dict):
        """ [REPLACE]
            Can be used to get info at each step - when collecting data
            can be used for e.g. getting data from info
        """
        pass

    def _collect_experiences_finished(self) -> dict:
        """ [REPLACE]
            Run at the end of _collect_experience:
            Can be used for e.g. adding extra logs obtained from info
        """
        return dict()

    def get_logs_config(self) -> Tuple[List[LogCfg], List[LogCfg]]:
        """
            Return headers configurations for logging master
        """
        return self._logs_train_config, self._logs_eval_config

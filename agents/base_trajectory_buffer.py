# Credits (inspired & adapted from) @ https://github.com/lcswillems/torch-rl

import torch
import collections
from typing import List

from utils.dictlist import DictList
from .base import AgentBase
from utils.data_primitives import Experience
from utils.vec_env import ParallelEnvWithLastObs


class TrajectoryBufferBase(AgentBase):
    """ The base class for the RL algorithms. """

    def __init__(self, *args, **kwargs):
        """
        cfg Should contain the following (at least for the base):
            :cfg max_trajectory_history
        """
        super().__init__(*args, parallel_env_class=ParallelEnvWithLastObs, **kwargs)

        cfg = self.cfg

        # -- Store helpers values
        self.max_trajectory_history = cfg.max_trajectory_history

        # Get from base
        recurrent = self.recurrent
        num_procs = self.num_procs
        device = self.device

        # -- Initialize experience data holders
        self.obs = self.env.reset()
        self.obs = self.preprocess_obss(self.obs, device=device)

        self.mask = torch.ones(num_procs, device=device)

        if recurrent:
            # Get memory size from model
            self.memory = torch.zeros(num_procs, self.model.memory_size, device=device)

        self._trajectories_buffer = collections.deque(maxlen=self.max_trajectory_history)
        self._envs_buffer = [[] for _ in range(num_procs)]
        self._trajectories_cnt = 0

        # -- Init evaluation environments and storage data
        if self.has_evaluator:
            eval_envs = self.eval_envs
            obs = eval_envs.reset()

    def update_parameters(self) -> dict:
        """  [REPLACE] Implement agent training  """

        # -- Collect experiences
        exps, logs = self._collect_experiences()

        # -- Train Loop
        # ...

        return logs

    def _collect_experiences(self):
        """ Run 1 step. Collects rollouts. """
        model = self.model
        preprocess_obss = self.preprocess_obss
        device = self.device
        recurrent = self.recurrent

        self.log_done_counter = 0
        self.log_return = log_return = []
        self.log_num_frames = log_num_frames = []

        finished_trajectories = []

        # ==========================================================================================
        # -- Run loop
        preprocessed_obs = self.obs
        with torch.no_grad():
            if recurrent:
                res_m = model(
                    preprocessed_obs, mask=self.mask,
                    memory=self.memory * self.mask.unsqueeze(1)
                )
                action, act_log_prob, value, memory = res_m["actions"], \
                                                      res_m["act_log_probs"], \
                                                      res_m["values"], \
                                                      res_m["memory"]
            else:
                res_m = model(preprocessed_obs, mask=self.mask)
                action, act_log_prob, value = res_m["actions"], \
                                              res_m["act_log_probs"], \
                                              res_m["values"]

        obss, rewards, dones, infos = self.env.step(action.cpu().numpy())
        self.obs = obss = preprocess_obss(obss, device=device)

        # -- Process other useful information each step
        self._collect_experiences_step(0, obss, rewards, dones, infos, preprocessed_obs, res_m)

        res_ms = DictList(res_m)
        for ix in range(len(dones)):
            step_datas = rewards[ix], dones[ix], infos[ix], obss[ix]
            step_data = Experience(preprocessed_obs[ix], *step_datas, res_ms[ix])
            self._envs_buffer[ix].append(step_data)

            if step_data.done:
                finished_trajectories.append(self._add_new_trajectory(self._envs_buffer[ix], True))
                self._envs_buffer[ix] = []

        return finished_trajectories, dict()

    def _add_new_trajectory(self, trajectory: List[Experience], env_done: bool = False) \
            -> List[Experience]:

        self._trajectories_buffer.append(trajectory)
        self._trajectories_cnt += 1
        return trajectory

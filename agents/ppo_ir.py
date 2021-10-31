# Credits (inspired & adapted from) @ https://github.com/lcswillems/torch-rl

from typing import List, Tuple
from utils.dictlist import DictList

from agents.ppo import PPO
from utils.logging_utils import LogCfg


class PPOir(PPO):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._logs_train_config += [
            LogCfg("task_success", True, "a", "TrR:u,", ":.2f", True, True)
        ]
        self.log_task_success = list()

    def _collect_experiences_step(self, frame_id: int, obs: List[DictList], reward: List[float],
                                  done: List[bool], info: List[dict], prev_obs: DictList,
                                  model_result: dict):

        for ix in range(len(done)):
            if done[ix]:
                self.log_task_success.append(int(info[ix]["full_task_achieved"]))

    def _collect_experiences(self) -> Tuple[DictList, dict]:
        self.log_task_success = list()

        exps, log = super()._collect_experiences()

        log["task_success"] = self.log_task_success
        return exps, log



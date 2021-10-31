from argparse import Namespace
from utils.dictlist import DictList


class Experience(Namespace):
    def __init__(self, obs: DictList, reward: float, done: bool, info: dict, next_obs: DictList,
                 model_result: DictList):
        super().__init__()

        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info
        self.next_obs = next_obs
        self.model_result = model_result

    def items(self):
        data = dict({
            "obs": dict(self.obs.items()),
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
            "next_obs": self.next_obs,
            "model_result": None if self.model_result is None else dict(self.model_result.items()),
        })
        return data.items()

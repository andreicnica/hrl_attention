import numpy as np
from argparse import Namespace
from typing import List
import torch
from gym_minigrid.minigrid import OBJECT_TO_IDX

from utils.data_primitives import Experience


class IntentCompletionBase:
    def __init__(self, cfg: Namespace):
        self.cfg = cfg

    @property
    def num_intents(self) -> int:
        raise NotImplementedError

    def intents_completion(self, trajectory: List[Experience]) -> List[float]:
        raise NotImplementedError


class IntentsChangeRoom(IntentCompletionBase):
    _change_room = torch.tensor([
        [-1, 0],
        [0, -1],
        [1, 0],
        [0, 1],
    ])

    def __init__(self, cfg: Namespace):
        super().__init__(cfg)
        self._num_intents = len(self._change_room)
        self._change_room = self._change_room.to(cfg.device)

    @property
    def num_intents(self):
        return self._num_intents

    def intents_completion(self, trajectory: List[Experience]):

        diff_room = trajectory[-1].obs.room_pos - trajectory[0].obs.room_pos
        change_id = torch.where((self._change_room == diff_room).all(axis=1))[0]

        rr = np.zeros(self._num_intents)

        if len(change_id) != 0 and not trajectory[-1].obs.margin:
            rr[change_id[0].item()] = 1.

        return rr


class IntentsChangePosition(IntentCompletionBase):
    _change_position = torch.tensor([
        [-1, 0],
        [0, -1],
        [1, 0],
        [0, 1],
    ])

    def __init__(self, cfg: Namespace):
        super().__init__(cfg)
        self._num_intents = len(self._change_position)
        self._change_position = self._change_position.to(cfg.device)

    @property
    def num_intents(self):
        return self._num_intents

    def intents_completion(self, trajectory: List[Experience]):

        diff_position = trajectory[-1].obs.agent_pos - trajectory[0].obs.agent_pos
        change_id = torch.where((self._change_position == diff_position).all(axis=1))[0]

        rr = np.zeros(self._num_intents)

        if len(change_id) != 0:
            rr[change_id[0].item()] = 1.

        return rr


class IntentsMultiTask(IntentCompletionBase):
    _ball_id = OBJECT_TO_IDX["ball"]
    _box_id = OBJECT_TO_IDX["box"]
    _key_id = OBJECT_TO_IDX["key"]

    def __init__(self, cfg: Namespace):
        super().__init__(cfg)
        self._num_intents = 4

    @property
    def num_intents(self):
        return self._num_intents

    def intents_completion(self, trajectory: List[Experience]):
        rr = np.zeros(self.num_intents)
        t0_obs, tf_obs = trajectory[0].obs, trajectory[-1].obs

        # Unblock door
        rr[0] = float((t0_obs.blocked == 1 and tf_obs.blocked == 0).item())

        # Pick Key
        rr[1] = float((t0_obs.carrying != self._key_id and tf_obs.carrying == self._key_id).item())

        # Open door
        rr[2] = float((t0_obs.door_is_open == 0 and tf_obs.door_is_open == 1).item())

        # Pick box
        rr[3] = float((t0_obs.carrying != self._box_id and tf_obs.carrying == self._box_id).item())

        return rr


class IntentsMultiObjects(IntentCompletionBase):
    def __init__(self, cfg: Namespace):
        super().__init__(cfg)
        self._num_intents = cfg.intent_size

    @property
    def num_intents(self):
        return self._num_intents

    def intents_completion(self, trajectory: List[Experience]):
        rr = np.zeros(self.num_intents)

        collected = trajectory[-1].obs.collected

        if collected != -1:
            rr[collected] = 1

        return rr



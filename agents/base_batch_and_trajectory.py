from typing import List, Any, Tuple
import collections

from utils.dictlist import DictList
from .base_batch import BatchBase
from utils.data_primitives import Experience
from utils.vec_env import ParallelEnvWithLastObs, ParallelEnvWithLastObsIndex


class TrajectoryBufferWithBatchBase(BatchBase):
    """ The base class for the RL algorithms. """

    def __init__(self, *args, **kwargs):

        if hasattr(args[0], "parallel_env_class"):
            if args[0].parallel_env_class == "ParallelEnvWithLastObsIndex":
                parallel_env_class = ParallelEnvWithLastObsIndex
            else:
                raise NotImplementedError
        else:
            parallel_env_class = ParallelEnvWithLastObs

        super().__init__(*args, parallel_env_class=parallel_env_class, **kwargs)

        self.max_trajectory_history = self.cfg.max_trajectory_history

        # Add also trajectory buffer
        self._trajectories_buffer = collections.deque(maxlen=self.max_trajectory_history)
        self._envs_buffer = [[] for _ in range(self.num_procs)]
        self._trajectories_cnt = 0
        self._crt_finished_trajectories = []
        self._crt_finished_trajectories_info = []  # Info about env id and @ which step if finished

    def update_parameters(self) -> dict:
        """  [REPLACE] Implement agent training  """

        # -- Collect experiences
        exps, logs = self._collect_experiences()

        # -- Train Loop
        # ...

        return logs

    def _collect_experiences_step(self, frame_id: int, obs: List[DictList], reward: List[float],
                                  done: List[bool], info: List[dict], prev_obs: DictList,
                                  model_result: dict):

        res_ms = DictList(model_result)
        for ix in range(len(done)):
            step_datas = reward[ix], done[ix], info[ix], obs[ix]
            step_data = Experience(prev_obs[ix], *step_datas, res_ms[ix])
            self._envs_buffer[ix].append(step_data)

            if step_data.done:
                self._crt_finished_trajectories.append(
                    self._add_new_trajectory(self._envs_buffer[ix], env_done=True)
                )
                # Add proxy info
                self._crt_finished_trajectories_info.append(dict({"proc": ix, "frame": frame_id}))
                self._envs_buffer[ix] = []

    def _add_new_trajectory(self, trajectory: List[Experience], env_done: bool = False) \
            -> List[Experience]:

        self._trajectories_buffer.append(trajectory)
        self._trajectories_cnt += 1
        return trajectory

    def _collect_experiences(self) -> Tuple[DictList, List[Experience], List[Any], dict]:
        """ Run 1 step. Collects rollouts and computes advantages. """
        self._crt_finished_trajectories.clear()
        self._crt_finished_trajectories_info.clear()

        exps, logs = super()._collect_experiences()

        return exps, self._crt_finished_trajectories, self._crt_finished_trajectories_info, logs

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import itertools


def get_goal_pos(goal_pos, size, max_offset):
    goal_pos_pos = np.array(list(itertools.product(range(-max_offset, max_offset + 1),
                                                   range(-max_offset, max_offset + 1))))
    goal_pos_pos[:, 0] += goal_pos[0]
    goal_pos_pos[:, 1] += goal_pos[1]

    # Filter out unavailable positions
    fff = np.all((goal_pos_pos >= 1) & (goal_pos_pos < (size - 1)), axis=1)
    goal_pos_pos = goal_pos_pos[fff]
    goal_pos_pos = goal_pos_pos[(goal_pos_pos != np.array([1, 1])).any(axis=1)]
    return goal_pos_pos


class EmptySeqEnvV0(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=16,
        agent_pos=(1, 1),
        agent_dir=None,
        goal_pos=(8, 8),
        goal_rand_offset=1,
        train=True,
        rand_generator=123,
        task_size=1,
        switch_steps=1024,
        fixed_batch=True,
    ):
        self.agent_start_pos = agent_pos
        self.agent_start_dir = None if agent_dir is None else np.clip(agent_dir, 0, 4)
        self.goal_start_pos = goal_pos
        self.goal_rand_offset = goal_rand_offset
        self.unwrapped.train = train
        self.unwrapped.eval_id = None

        goal_poss = np.array(list(itertools.product(
            range(-goal_rand_offset, goal_rand_offset + 1),
            range(-goal_rand_offset, goal_rand_offset + 1)
        )))
        goal_poss[:, 0] += goal_pos[0]
        goal_poss[:, 1] += goal_pos[1]

        # Filter out unavailable positions
        fff = np.all((goal_poss >= 1) & (goal_poss < (size - 1)), axis=1)
        goal_poss = goal_poss[fff]
        goal_poss = goal_poss[(goal_poss != np.array([1, 1])).any(axis=1)]

        self.goal_set_rstate = np.random.RandomState(rand_generator)
        self.goal_set_rstate.shuffle(goal_poss)

        self._crt_ep_step = 0
        self._crt_step = 0
        self._crt_goal_batch = None
        self._crt_goal_batch_idx = 0

        self.batch_size = task_size
        self.switch_steps = switch_steps

        if fixed_batch:
            self.train_goals = goal_poss[:(len(goal_poss) // task_size) * task_size]
        else:
            self.train_goals = goal_poss

        self._prev_reset = -1
        self.next_batch()

        super().__init__(
            grid_size=size,
            max_steps=400,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    @property
    def step_count(self):
        return self._crt_ep_step

    @step_count.setter
    def step_count(self, value):
        self._crt_step += 1
        self._crt_ep_step = value

    def next_batch(self):
        if self._prev_reset == self._crt_step:
            return

        train_goals = self.train_goals
        batch_size = self.batch_size
        self._crt_goal_batch_idx = (self._crt_goal_batch_idx + batch_size) % len(train_goals)
        btch_start_idx = self._crt_goal_batch_idx
        if btch_start_idx + batch_size > len(train_goals):
            rest = (btch_start_idx + batch_size) % len(train_goals)
            goals_batch = np.concatenate([train_goals[btch_start_idx:], train_goals[:rest]])
        else:
            goals_batch = train_goals[btch_start_idx: btch_start_idx + batch_size]

        self._crt_goal_batch = goals_batch
        self._prev_reset = self._crt_step

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.goal_start_pos is not None:
            goal = Goal()
            goal_pos = self.goal_start_pos
            rnd_off = self.goal_rand_offset

            if rnd_off is not None:
                if self.unwrapped.train:
                    crt_batch = self._crt_goal_batch
                    goal_pos = crt_batch[np.random.randint(len(crt_batch))].tolist()
                else:
                    eval_id = self.unwrapped.eval_id

                    if eval_id is None:
                        goal_pos = self.train_goals[np.random.randint(len(self.train_goals))]
                        goal_pos = goal_pos.tolist()
                    else:
                        goal_pos = self.train_goals[eval_id].tolist()

            self.put_obj(goal, *goal_pos)
            goal.init_pos, goal.cur_pos = goal_pos
        else:
            # Place a goal square in the bottom-right corner
            goal_pos = [width - 2, height - 2]
            self.put_obj(Goal(), *goal_pos)

        self.unwrapped._crt_goal_pos = goal_pos

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            if self.agent_start_dir is None:
                self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self.agent_start_dir is not None:
            self.agent_dir = self.agent_start_dir

        self.mission = "get to the green goal square"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info

    def _get_done(self):
        done = False

        if self.step_count >= self.max_steps:
            done = True

        return done

register(
    id='MiniGrid-EmptySeq-v0',
    entry_point='gym_minigrid.envs:EmptySeqEnvV0'
)

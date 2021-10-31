import gym
import numpy as np
from gym_minigrid.wrappers import FullyObsWrapper, OneHotPartialObsWrapper, CustomOneHotPartialObsWrapper
from gym_minigrid.minigrid import OBJECT_TO_IDX
from gym import spaces


from utils.intent_completion import IntentsChangeRoom


class ExtendRoomFullyObs(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["agent_pos"] = spaces.Box(
            low=0, high=255, shape=(2, ), dtype='uint8'
        )
        self.observation_space.spaces["room_pos"] = spaces.Box(
            low=0, high=255, shape=(2, ), dtype='uint8'
        )
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()

        carrying = 0 if env.carrying is None else OBJECT_TO_IDX[env.carrying.type]
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            carrying,
            env.agent_dir
        ])

        agent_pos = self.agent_pos
        room_pos = agent_pos // (self.room_size - 1)
        goal_room = self.goal_crt_pos // (self.room_size - 1)
        margind = agent_pos % (self.room_size - 1)

        if np.any(margind < 2) or np.any(margind >= self.room_size - 2):
            margin = True
        else:
            margin = False

        return {
            'agent_pos': agent_pos,
            'room_pos': room_pos,
            'goal_room': goal_room,
            'image': full_grid,
            'margin': margin,
        }


class ExtendRoomFullyObsExtra(ExtendRoomFullyObs):
    def observation(self, obs):
        env = self.unwrapped
        obs = super().observation(obs)
        room_pos = obs["room_pos"]

        room_size = self.room_size - 1
        center_room = room_pos * room_size + room_size // 2

        available_intents = np.zeros((4), dtype=np.bool)
        for dir in range(4):
            op_act = IntentsChangeRoom._change_room[dir].numpy()
            door_pos = center_room + op_act * (room_size // 2)
            elem = obs["image"][door_pos[0], door_pos[1]]
            if elem[0] == 1:
                available_intents[dir] = True
        obs['affordable'] = available_intents
        return obs


class ExtendBlockedFullyObs(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["carrying"] = spaces.Box(
            low=0, high=255, shape=(1, ), dtype='uint8'
        )
        self.observation_space.spaces["blocked"] = spaces.Box(
            low=0, high=1, shape=(1, ), dtype='bool'
        )
        self.observation_space.spaces["door_is_open"] = spaces.Box(
            low=0, high=1, shape=(1, ), dtype='bool'
        )
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()

        carrying = 0 if env.carrying is None else OBJECT_TO_IDX[env.carrying.type]
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            carrying,
            env.agent_dir
        ])

        door_is_open = env.door.is_open

        # Check if door is blocked
        ooo = self.grid.get(*self.unwrapped.blocked_pos)
        if ooo is None:
            blocked = False
        else:
            blocked = True

        return {
            'blocked': blocked,
            'carrying': carrying,
            'door_is_open': door_is_open,
            'image': full_grid
        }


class ExtendBlockedObs(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["carrying"] = spaces.Box(
            low=0, high=255, shape=(1, ), dtype='uint8'
        )
        self.observation_space.spaces["blocked"] = spaces.Box(
            low=0, high=1, shape=(1, ), dtype='bool'
        )
        self.observation_space.spaces["door_is_open"] = spaces.Box(
            low=0, high=1, shape=(1, ), dtype='bool'
        )

    def observation(self, obs):
        env = self.unwrapped

        carrying = 0 if env.carrying is None else OBJECT_TO_IDX[env.carrying.type]

        door_is_open = env.door.is_open

        # Check if door is blocked
        ooo = self.grid.get(*self.unwrapped.blocked_pos)
        if ooo is None:
            blocked = False
        else:
            blocked = True

        return {
            'blocked': blocked,
            'carrying': carrying,
            'door_is_open': door_is_open,
            'image': obs["image"]
        }


class ExtendMultiObjFullObs(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

        self.observation_space.spaces["carrying"] = spaces.Box(
            low=0, high=255, shape=(1, ), dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()

        carrying = 0 if env.carrying is None else OBJECT_TO_IDX[env.carrying.type]
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            carrying,
            env.agent_dir
        ])
        collected = obs["collected"]

        return {
            'image': full_grid,
            'collected': collected
        }


class FreeMove(gym.core.Wrapper):
    _move_actions = np.array([
        [-1, 0],
        [0, -1],
        [1, 0],
        [0, 1],
    ])
    _move_actions_list = _move_actions.tolist()

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        env = self.unwrapped
        env.step_count += 1

        reward = 0
        done = False

        # Move action
        if 0 <= action < 4:
            # Get the new possible position for the action
            fwd_pos = self.agent_pos + self._move_actions[action]

            # Get the contents of the cell in front of the agent
            fwd_cell = env.grid.get(*fwd_pos)

            if fwd_cell is None or fwd_cell.can_overlap():
                env.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == 'goal':
                done = True
                reward = env._reward()
            if fwd_cell is not None and fwd_cell.type == 'lava':
                done = True

        if env.step_count >= env.max_steps:
            done = True

        obs = env.gen_obs()
        info = {}

        return obs, reward, done, info


class MultiTaskMultiObj(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.unwrapped._fixed_task_id = 0
        self.new_task = False

    def step(self, action):
        if action == 999:
            num_tasks = self.unwrapped._num_tasks
            self.unwrapped._fixed_task_id = (self.unwrapped._fixed_task_id + 1) % num_tasks
            self.new_task = True
            return None, None, None, None

        obs, reward, done, info = super().step(action)
        if self.new_task:
            done = True
            self.new_task = False

        return obs, reward, done, info


class MultiTaskMultiObjShuffleOrderAll(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.new_task = False

    def step(self, action):
        if action == 999:
            self.unwrapped.get_new_sequence_all()
            self.new_task = True
            return None, None, None, None

        obs, reward, done, info = super().step(action)
        if self.new_task:
            done = True
            self.new_task = False

        return obs, reward, done, info


class MultiTaskMultiObjShuffleOrderTask(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.new_task = False

    def step(self, action):
        if action == 999:
            self.unwrapped.get_new_sequence_per_task()
            self.new_task = True
            return None, None, None, None

        obs, reward, done, info = super().step(action)
        if self.new_task:
            done = True
            self.new_task = False

        return obs, reward, done, info


class FreeMoveGridRooms(FreeMove):
    def __init__(self, env):
        super().__init__(env)
        self._intents = IntentsChangeRoom._change_room.numpy()
        self._delayed_done = False
        if self.unwrapped._multitask:
            assert self.unwrapped._goal_rooms is None, "Wrong config for change rooms"
            self._change_rooms = [
                [3, 3], [9, 3], [15, 3],
                [3, 9], [9, 9], [15, 9],
                [3, 15], [9, 15], [15, 15],
            ]
            self.rnd_state = 0
            self._change_room_id = 0
            self.change_task()
        self.new_task = False

    def step(self, action: int):
        if action == 999:
            self.change_task()
            self.new_task = True
            return None, None, None, None

        env = self.unwrapped
        env.step_count += 1

        new_op = None
        reward = 0
        done = False
        info = dict()

        if action > 10:
            intent = action // 100 - 1
            new_op = (action // 10) % 10
            action = action % 10
            if new_op:
                env._intent_start_room = env.agent_pos // (env.room_size - 1)
                env._intent = intent

        margind = self.agent_pos % (env.room_size - 1)

        # Move action
        if 0 <= action < 4 and not self._delayed_done:
            # Get the new possible position for the action
            fwd_pos = self.agent_pos + self._move_actions[action]

            # Get the contents of the cell in front of the agent
            fwd_cell = env.grid.get(*fwd_pos)

            if fwd_cell is None or fwd_cell.can_overlap():
                env.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == 'goal':
                done = True
                reward = env._reward()
            if fwd_cell is not None and fwd_cell.type == 'lava':
                done = True

        if self._delayed_done and new_op != 1.:
            done = True
            self._delayed_done = False
        else:
            self._delayed_done = False
            if env._reset_on_intent and env._intent_start_room is not None:
                crt_room = self.agent_pos // (env.room_size - 1)
                room_change = crt_room - env._intent_start_room

                if np.any(margind < 2) or np.any(margind >= env.room_size - 2):
                    margin = True
                else:
                    margin = False

                # Collect last <intent_window> intent completions
                env._intent_completions.append(np.all(self._intents[env._intent] == room_change))
                reset_done = np.all(env._intent_completions) and not margin
                done = self._delayed_done
                self._delayed_done = reset_done

        # Reward upon reaching room
        if env._reward_room and not env._fake_goal:
            room_pos = self.agent_pos // (env.room_size - 1)
            goal_room = self.goal_crt_pos // (env.room_size - 1)

            # if np.any(margind < 1) or np.any(margind >= env.room_size - 1):
            #     margin = True
            # else:
            #     margin = False

            if (room_pos == goal_room).all():
                done = True
                reward = env._reward()

        if env.step_count >= env.max_steps:
            done = True

        if self.new_task:
            done = True
            self.new_task = False

        obs = env.gen_obs()

        return obs, reward, done, info

    def reset(self, **kwargs):
        self._delayed_done = False
        return super().reset(**kwargs)

    def change_task(self):
        self._change_room_id += 1
        if self._change_room_id >= len(self._change_rooms):
            self._change_room_id = 0
            self.rnd_state += 1
            # np.random.RandomState(self.rnd_state).shuffle(self._change_rooms)
        self.unwrapped._goal_default_pos = self._change_rooms[self._change_room_id]


class FreeMove(gym.core.Wrapper):
    _move_actions = np.array([
        [-1, 0],
        [0, -1],
        [1, 0],
        [0, 1],
    ])
    _move_actions_list = _move_actions.tolist()

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        env = self.unwrapped
        env.step_count += 1

        reward = 0
        done = False

        # Move action
        if 0 <= action < 4:
            # Get the new possible position for the action
            fwd_pos = self.agent_pos + self._move_actions[action]

            # Get the contents of the cell in front of the agent
            fwd_cell = env.grid.get(*fwd_pos)

            if fwd_cell is None or fwd_cell.can_overlap():
                env.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == 'goal':
                done = True
                reward = env._reward()
            if fwd_cell is not None and fwd_cell.type == 'lava':
                done = True

        if env.step_count >= env.max_steps:
            done = True

        obs = env.gen_obs()
        info = {}

        return obs, reward, done, info


class FetchWrapper(gym.core.Wrapper):

    def __init__(self, env, flatten=('observation', 'desired_goal')):
        env.env.reward_type = "dense"

        super().__init__(env)

        obs_space = env.observation_space
        low = np.concatenate([obs_space[x].low for x in flatten])
        high = np.concatenate([obs_space[x].high for x in flatten])

        shp = np.sum([obs_space[x].shape[0] for x in flatten])

        self.observation_space.spaces["obs"] = spaces.Box(
            low=low[0], high=high[0], shape=(shp, ), dtype='float32'
        )

        self.flatten = flatten
        # Create the suitable environment

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs["obs"] = np.concatenate([obs[x] for x in self.flatten])
        if info["is_success"] == 1.:
            done = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        obs["obs"] = np.concatenate([obs[x] for x in self.flatten])

        return obs

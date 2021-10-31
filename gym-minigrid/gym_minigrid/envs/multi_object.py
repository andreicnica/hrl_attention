import math
import copy

from gym_minigrid.minigrid import COLOR_NAMES, Box, Ball, Key
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register
from gym_minigrid.minigrid import *

OBJ_TYPES = [Box, Ball, Key]


class MultiObject(RoomGrid):
    """

    """

    def __init__(self,
                 seed=None,
                 full_task=True,
                 room_size=7,
                 see_through_walls=True,
                 task_size=3,
                 num_tasks=1,
                 task_id=1,
                 reward_pickup=False,
                 **kwargs
                 ):

        self.full_task = full_task
        self._randPos = self._rand_pos
        self._carrying = None
        self._task_size = task_size
        self._num_tasks = num_tasks
        self._reward_pickup = reward_pickup

        self._partial_reward = 0.5 / float(task_size)
        self._num_obj = num_obj = task_size * num_tasks
        self._task_id = task_id
        self._rand_state = rnd = np.random.RandomState(task_id)

        self._collected_obj = list()
        self._available_obj = list([False] * self._num_obj)
        self._obj_pos = list()
        self._crt_task = None
        self._fixed_task_id = None

        self._objs = self.get_obj_list()

        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed,
            **kwargs
        )
        self.see_through_walls = see_through_walls
        self.mission = "use the key to open the door and then get to the goal"

        self.observation_space.spaces["carrying"] = spaces.Box(
            low=0, high=255, shape=(1, ), dtype='uint8'
        )

    def get_obj_list(self):
        self._rand_state = rnd = np.random.RandomState(self._task_id)
        objs = []

        while len(objs) < self._num_obj:
            obj = (rnd.choice(OBJ_TYPES), rnd.choice(COLOR_NAMES))
            if obj not in objs:
                objs.append(obj)
        return objs

    def get_new_sequence_all(self):
        # Shuffle all objects
        old_seq = copy.deepcopy(self._objs)
        while old_seq == self._objs:
            self._rand_state.shuffle(self._objs)

    def get_new_sequence_per_task(self):
        # Shuffle objs within the task
        task_size = self._task_size
        objs = self._objs
        new_objs = []

        for itask in range(self._num_tasks):
            st_i = itask * task_size
            task_objs = objs[st_i: st_i + task_size]
            new_task_objs = copy.deepcopy(task_objs)

            while new_task_objs == task_objs:
                self._rand_state.shuffle(new_task_objs)
            new_objs += new_task_objs
        self._objs = new_objs

    def reset(self):
        self._crt_task = None
        self._collected_obj = list()
        self._available_obj = list([False] * self._num_obj)
        self._obj_pos = list()

        obs = super().reset()
        obs["collected"] = -1
        obs["available_obj"] = self._available_obj

        return obs

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        num_tasks = self._num_tasks
        task_size = self._task_size
        objs = self._objs

        if self._fixed_task_id is None:
            self._crt_task = task = self.np_random.randint(num_tasks)
        else:
            self._crt_task = task = self._fixed_task_id

        st_i = task * task_size
        self._crt_task_ids = list(range(st_i, st_i + task_size))

        for i in range(task_size):
            task_obj = objs[st_i + i]
            obj = task_obj[0](task_obj[1])
            obj._obj_id = st_i + i
            pos = self.place_obj(obj)
            self._obj_pos.append(pos)
            self._available_obj[obj._obj_id] = True

        self.place_agent()

    def step(self, action):
        self.carrying = None

        obs, reward, done, info = super().step(action)
        reward = 0

        obs["collected"] = -1
        info["full_task_achieved"] = False

        if self.carrying is not None:
            obj_id = self.carrying._obj_id
            self._collected_obj.append(obj_id)
            # obs = self.gen_obs()
            obs["collected"] = obj_id
            self._available_obj[obj_id] = False
            if self._reward_pickup:
                reward += self._partial_reward

        if len(self._collected_obj) == self._task_size:
            if self.full_task and self._collected_obj == self._crt_task_ids:
                reward = +1
                info["full_task_achieved"] = True
            done = True

        obs["available_obj"] = self._available_obj

        return obs, reward, done, info


class MultiObjectEGO(MultiObject):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, *args, room_size=14, **kwargs):
        self.agent_view_size = agent_view_size = (room_size - 3) * 2 + 1
        super().__init__(*args, agent_view_size = agent_view_size, room_size=room_size, **kwargs)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

    def get_view_exts(self):
        topY = self.agent_pos[1] - self.agent_view_size // 2
        topX = self.agent_pos[0] - self.agent_view_size // 2
        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size

        return (topX, topY, botX, botY)

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2 ,
                                                   self.agent_view_size // 2))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height // 2
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        obs = super().gen_obs()
        w, h, _ = obs["image"].shape
        agent_pos = w // 2, h // 2
        obs["image"][agent_pos[0], agent_pos[1], 2] = 2
        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)
        vis_mask.fill(True)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size // 2),
            agent_dir=3,
            highlight_mask=vis_mask
        )

        return img

class MultiObjectEGOOneHot(MultiObjectEGO):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        obs_shape = self.observation_space['image'].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], num_bits),
            dtype='uint8'
        )

    def observation(self, obs):
        img = obs['image']
        out = np.zeros(self.observation_space.shape, dtype='uint8')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        obs["image"] = out
        return obs


class MultiObjectEGOtest(MultiObjectEGO):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def step(self, complex_actions):

        op = (complex_actions // 10) - 1
        action = complex_actions % 10

        # if not self._available_obj[op]:
        #     obs = self.gen_obs()
        #     return obs, 0, True, dict()

        obs, reward, done, info = super().step(action)

        obs["collected"] = -1

        if self.carrying is not None:
            obj_id = self.carrying._obj_id

            if obj_id == op:
                self._collected_obj.append(obj_id)
                self.carrying = None
                obs = self.gen_obs()
                obs["collected"] = obj_id
                self._available_obj[obj_id] = False
            else:
                pos = self.place_obj(self.carrying)
                self._obj_pos[obj_id] = pos
                self.carrying = None
                obs = self.gen_obs()

        if len(self._collected_obj) == self._task_size:
            if self.full_task and self._collected_obj == self._crt_task_ids:
                reward = 1
            done = True

        obs["available_obj"] = self._available_obj

        return obs, reward, done, info




register(
    id='MiniGrid-MultiObject-v0',
    entry_point='gym_minigrid.envs:MultiObject'
)

register(
    id='MiniGrid-MultiObjectEGO-v0',
    entry_point='gym_minigrid.envs:MultiObjectEGO'
)

register(
    id='MiniGrid-MultiObjectEGOtest-v0',
    entry_point='gym_minigrid.envs:MultiObjectEGOtest'
)

register(
    id='MiniGrid-MultiObjectEGOOneHot-v0',
    entry_point='gym_minigrid.envs:MultiObjectEGOOneHot'
)

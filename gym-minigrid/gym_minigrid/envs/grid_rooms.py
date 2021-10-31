from collections import deque
import numpy as np

from gym_minigrid.minigrid import *
from gym_minigrid.roomgrid import RoomGrid, Room
from gym_minigrid.register import register


class GridRooms(RoomGrid):
    def __init__(self,
                 num_rows=3,
                 num_cols=3,
                 seed=None,
                 max_steps=400,
                 agent_pos=None,
                 goal_pos=None,
                 room_size=7,
                 middle_door=True,
                 reset_on_intent=False,
                 intent_window=3,
                 goal_center_room=False,
                 fake_goal=False,
                 reward_room=True,
                 goal_rooms=0,
                 multitask=False
                 ):

        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        if middle_door:
            self._door_rand = lambda x, y: x+(y-x)//2
        else:
            self._door_rand = self._rand_int

        self._reset_on_intent = reset_on_intent
        self._intent_completions = deque([0] * intent_window, maxlen=intent_window)
        self._intent_start_room = None

        self._goal_center_room = goal_center_room
        self._fake_goal = fake_goal
        self._reward_room = reward_room
        self._multitask = multitask  # implemented in wrapper

        room_sets = [[[15, 15]], [[9, 15], [15, 15]], [[3, 15], [9, 15], [15, 15]]]
        self._goal_rooms = None

        if goal_rooms > 0:
            self._goal_rooms = room_sets[goal_rooms-1]

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            max_steps=max_steps,
            seed=seed
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.room_grid = []

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = Room(
                    (i * (self.room_size-1), j * (self.room_size-1)),
                    (self.room_size, self.room_size)
                )
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._door_rand(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._door_rand(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        self.mission = "get to the green goal square"
        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]
                for pos in room.door_pos:
                    if pos is None:
                        continue
                    self.grid.set(pos[0], pos[1], None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = np.array(self._agent_default_pos)
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        # Randomize the goal start position and orientation
        new_goal = Goal()
        if self._fake_goal:
            new_goal.type = "fake_goal"

        if self._goal_rooms is not None:
            self._goal_default_pos = self._goal_rooms[self._rand_int(0, len(self._goal_rooms))]

        if self._goal_default_pos is not None:
            if self._goal_default_pos[0] > 0:
                goal = new_goal
                goal_pos = self._goal_default_pos
                self.put_obj(goal, *goal_pos)
                goal.init_pos = goal.cur_pos = np.array(goal_pos)
                new_goal = goal
        else:
            if self._goal_center_room:
                room_size = self.room_size

                # Position goal in the center of one random room except for the one with the agent
                ag_room_pos = goal_room = tuple(self.agent_pos // (self.room_size - 1))

                while goal_room == ag_room_pos:
                    goal_room = (self._rand_int(0, self.num_cols), self._rand_int(0, self.num_rows))

                i = goal_room[0] * (room_size - 1) + room_size//2
                j = goal_room[1] * (room_size - 1) + room_size//2
                self.put_obj(new_goal, i, j)
                new_goal.init_pos = new_goal.cur_pos = np.array([i, j])
            else:
                self.place_obj(new_goal)

        if new_goal.cur_pos[0] == self.agent_pos[0] and new_goal.cur_pos[1] == self.agent_pos[1]:
            # move agent in empty pos
            i, j = new_goal.cur_pos
            for ni, nj in [[i-1, j], [i, j-1], [i+1, j], [i, j+1]]:
                c = self.grid.get(ni, nj)
                if c is None:
                    self.agent_pos = np.array([ni, nj])
                    self.grid.set(ni, nj, None)
                    break
        # print(self.agent_pos, new_goal.cur_pos)
        self.goal_crt_pos = new_goal.cur_pos

    def step(self, action):
        # The step method might be overwritten by a wrapper
        if action > 10:
            intent = action // 100 - 1
            new_op = (action // 10) % 10
            action = action % 10
            if new_op:
                self._intent_start_room = self.agent_pos // (self.room_size - 1)

        obs, reward, done, info = MiniGridEnv.step(self, action)
        if self._reward_room and not self._fake_goal:
            room_pos = self.agent_pos // (self.room_size - 1)
            goal_room = self.goal_crt_pos // (self.room_size - 1)
            if (room_pos == goal_room).all():
                done = True
                reward = self._reward()

        return obs, reward, done, info


class GridRooms4(GridRooms):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_cols=2, num_rows=2, **kwargs)


class GridRooms9(GridRooms):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_cols=3, num_rows=3, **kwargs)


register(
    id="MiniGrid-GridRooms-v0",
    entry_point="gym_minigrid.envs:GridRooms"
)

register(
    id="MiniGrid-GridRooms4-v0",
    entry_point="gym_minigrid.envs:GridRooms4"
)

register(
    id="MiniGrid-GridRooms9-v0",
    entry_point="gym_minigrid.envs:GridRooms9"
)
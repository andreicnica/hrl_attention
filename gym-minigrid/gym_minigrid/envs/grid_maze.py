from collections import deque
import numpy as np
import random

from gym_minigrid.minigrid import *
from gym_minigrid.envs.grid_rooms import GridRooms
from gym_minigrid.register import register

MOVE_VEC = [
    np.array([-1, 0]),
    np.array([0, -1]),
    np.array([1, 0]),
    np.array([0, 1]),
]


def connect_rooms(crt_rooms, goal_room, max_room):
    moves = random.sample(MOVE_VEC, len(MOVE_VEC))

    for im, move in enumerate(moves):
        lx, ly = crt_rooms[-1]
        new_x, new_y = lx + move[0], ly + move[1]
        if 0 <= new_x < max_room and 0 <= new_y < max_room:
            vroom = tuple([new_x, new_y])
            if vroom not in crt_rooms:
                crt_rooms.append(vroom)
                if vroom == goal_room:
                    return True, crt_rooms
                else:
                    reached, new_rooms = connect_rooms(crt_rooms, goal_room, max_room)
                    if reached:
                        return True, new_rooms
                crt_rooms.pop()
    return False, list()


def get_room_neighbour(rix, x, y):
    # Door positions, order is right, down, left, up
    # Indexing is (column, row)
    nx, ny = x, y
    if rix == 0:
        ny += 1
    elif rix == 1:
        nx += 1
    elif rix == 2:
        ny -= 1
    else:
        nx -= 1
    return tuple([nx, ny])


class GridMaze(GridRooms):
    def __init__(self,
                 grid_size=6,
                 goal_center_room=True,
                 close_doors_trials=0.2,
                 same_maze=True,
                 **kwargs
                 ):
        num_rows = num_cols = grid_size
        assert num_rows == num_cols, "Square only now"
        self.close_doors_trials = int(num_rows * num_rows * 4 * close_doors_trials)

        super().__init__(num_rows=num_rows, num_cols=num_cols, goal_center_room=goal_center_room,
                         **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        room_size = self.room_size - 1
        max_rooms = self.num_rows
        ag_pos = self.agent_pos
        close_doors_trials = self.close_doors_trials

        ag_start_room = ag_pos // room_size
        goal_room = tuple(self.goal_crt_pos // room_size)

        checked_rooms = [tuple(ag_start_room)]

        if goal_room[0] == ag_start_room[0] and goal_room[1] == ag_start_room[1]:
            reached, rooms = True, checked_rooms
        else:
            reached, rooms = connect_rooms(checked_rooms, goal_room, max_rooms)

        for i in range(close_doors_trials):
            # Pick random room. Try to close random door but not the one connecting ag to goal
            x, y = np.random.randint(0, max_rooms, (2,))
            room = self.room_grid[x][y]
            doors_i = [i for i in range(len(room.door_pos)) if room.door_pos[i] is not None]

            if len(doors_i) == 0:
                continue
            select_door = random.sample(doors_i, 1)[0]
            door_neigh_room = get_room_neighbour(select_door, x, y)

            can_remove_door = True
            # Door positions, order is right, down, left, up
            room_coord = tuple([x, y])
            if room_coord in rooms:
                # Check if door connects to closeby rooms
                rpos = rooms.index(room_coord)
                if rpos > 0 and rooms[rpos - 1] == door_neigh_room:
                    can_remove_door = False
                if rpos < len(rooms) - 1 and rooms[rpos + 1] == door_neigh_room:
                    can_remove_door = False
            if can_remove_door:
                dy, dx = room.door_pos[select_door]
                if dx != ag_pos[0] or dy != ag_pos[1]:
                    # dx, dy = room.door_pos[select_door]
                    self.grid.set(dx, dy, Wall())
                    room.door_pos[select_door] = None

class GridMazeEGO(GridMaze):
    def __init__(self, *args, agent_view_size=13, see_through_walls=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.see_through_walls = see_through_walls
        self.agent_view_size = agent_view_size
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

        # for i in range(self.agent_dir + 1):
        #     grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2,
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


register(
    id="MiniGrid-GridMaze-v0",
    entry_point="gym_minigrid.envs:GridMaze"
)

register(
    id="MiniGrid-GridMazeEGO-v0",
    entry_point="gym_minigrid.envs:GridMazeEGO"
)

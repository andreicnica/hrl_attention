import math

from gym_minigrid.minigrid import Ball
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register
from gym_minigrid.minigrid import *


class BlockedUnlockPickupV2(RoomGrid):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, seed=None, full_task=False, with_reward=False, num_rows=2,
                 reward_ball=False, see_through_walls=True, agent_view_size=7,
                 reset_on_intent=False):
        room_size = 6
        self.full_task = full_task
        self._randPos = self._rand_pos
        self._carrying = None
        self._with_reward = with_reward
        self._reward_ball = reward_ball
        self._reset_on_intent = reset_on_intent

        super().__init__(
            num_rows=num_rows,
            num_cols=2,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed,
        )
        self.see_through_walls = see_through_walls
        self.agent_view_size = agent_view_size
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self._carrying = None

        if self.full_task:
            self._full_task_gen_grid()
        else:
            self._all_initial_state()

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)

        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = self._carrying

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def _all_initial_state(self, ):
        rand_dir = True

        # self.place_agent(0, 0)
        obj, _ = self.add_object(1, 0, kind="box")
        self.unwrapped.box = obj

        # Make sure the two rooms are directly connected by a locked door
        door, door_pos = self.add_door(0, 0, 0, locked=True)
        # Open door
        door.is_open = self._rand_int(0, 2) == 0
        self.unwrapped.door = door

        # Block the door with a ball
        color = self._rand_color()

        self.unwrapped.blocked_pos = door_pos[0]-1, door_pos[1]

        # 50% blocked door by Ball
        the_ball = None
        if self._rand_int(0, 2) == 0:
            ball_pos = door_pos[0]-1, door_pos[1]
            the_ball = Ball(color)
            self.grid.set(door_pos[0]-1, door_pos[1], the_ball)
            blocked = True
        else:
            the_ball, ball_pos = self.add_object(self._rand_int(0, 2), 0, 'ball', color)
            blocked = False

        self.unwrapped.the_ball = the_ball
        self.unwrapped.the_ball.cur_pos = ball_pos

        # Place agent
        room_pos = self._rand_int(0, 2)
        room = self.get_room(room_pos, 0)

        not_placed = True
        while not_placed:
            new_pos = room.rand_pos(self)
            ooo = self.grid.get(*new_pos)

            if ooo is not None:
                if (ooo.type == "door" and door.is_open) or ooo.type == "ball":
                    self._carrying = ooo
                    self.grid.set(*new_pos, None)
                else:
                    continue

            not_placed = False

        self.agent_pos = new_pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        # Add a key to unlock the door
        if self._carrying is not None:
            room_pos_key = self._rand_int(0, 2) if door.is_open else room_pos
            key, _ = self.add_object(room_pos_key, 0, 'key', door.color)
        else:
            # Should place key anywhere - even "over" agent
            # Must also consider the door if it is locked
            if not door.is_open and not blocked and self._rand_int(0, 4) == 0:
                # TODO hardcoded prob for having key
                self._carrying = key = Key(door.color)
            else:
                room_pos_key = self._rand_int(0, 2) if door.is_open else room_pos
                room = self.get_room(room_pos_key, 0)

                not_placed = True
                while not_placed:
                    new_pos = room.rand_pos(self)
                    ooo = self.grid.get(*new_pos)
                    key = Key(door.color)
                    if ooo is not None or new_pos == self.agent_pos:
                        if new_pos == self.agent_pos:
                            self._carrying = key
                        else:
                            continue
                    else:
                        self.grid.set(*new_pos, key)
                        key.cur_pos = new_pos
                    not_placed = False

        self.unwrapped.key = key

        self.obj = obj if not self._reward_ball else the_ball
        self.mission = "pick up the %s %s" % (obj.color, obj.type)

    def _full_task_gen_grid(self, ):
        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        self.unwrapped.box = obj

        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=True)
        self.unwrapped.door = door
        self.unwrapped.blocked_pos = pos[0]-1, pos[1]

        # Block the door with a ball
        color = self._rand_color()
        the_ball = Ball(color)
        self.unwrapped.the_ball = the_ball
        self.unwrapped.the_ball.cur_pos = pos[0]-1, pos[1]

        self.grid.set(pos[0]-1, pos[1], the_ball)
        the_ball.cur_pos = [pos[0]-1, pos[1]]
        # Add a key to unlock the door
        key, _ = self.add_object(0, 0, 'key', door.color)
        self.unwrapped.key = key

        self.place_agent(0, 0)

        self.obj = obj if not self._reward_ball else the_ball
        self.mission = "pick up the %s %s" % (obj.color, obj.type)

    def step(self, action):
        ooo = self.grid.get(*self.unwrapped.blocked_pos)
        door_open = self.unwrapped.door.is_open
        intent = None

        if action >= 10:
            intent = action // 100 - 1
            new_op = (action // 10) % 10
            action = action % 10

        obs, reward, done, info = super().step(action)

        if self._reset_on_intent:
            if action == self.actions.pickup:
                if self.carrying == self.unwrapped.box and (intent is None or intent == 3):
                    done = True
                elif self.carrying == self.unwrapped.key and (intent is None or intent == 1):
                    done = True
            if ooo == self.unwrapped.the_ball and self.grid.get(*self.unwrapped.blocked_pos) is None:
                if intent is None or intent == 0:
                    done = True
            if not door_open and self.unwrapped.door.is_open:
                if intent is None or intent == 2:
                    done = True

        if self._with_reward:
            if action == self.actions.pickup:
                if self.carrying and self.carrying == self.obj:
                    reward = self._reward()
                    done = True

        return obs, reward, done, info


class BlockedUnlockPickupEGOV3(BlockedUnlockPickupV2):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, *args, agent_view_size=17, **kwargs):
        super().__init__(*args, agent_view_size=agent_view_size, **kwargs)

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
    id='MiniGrid-BlockedUnlockPickup-v2',
    entry_point='gym_minigrid.envs:BlockedUnlockPickupV2'
)


register(
    id='MiniGrid-BlockedUnlockPickupEGO-v0',
    entry_point='gym_minigrid.envs:BlockedUnlockPickupEGOV3'
)

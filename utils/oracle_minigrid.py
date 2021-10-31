import numpy as np

DIR_INDEX = dict({
    (1, 0): 0,
    (0, 1): 1,
    (-1, 0): 2,
    (0, -1): 3
})

trycnt = 0


def oracle_act(agpos, agdir, solution_coords):
    if not np.all(agpos == solution_coords[0]):
        return False, -1, solution_coords

    diff = solution_coords[1] - agpos
    tgtdir = DIR_INDEX[tuple(diff)]

    if agdir == tgtdir:
        if len(solution_coords) == 2:
            return True, 3, solution_coords[1:]
        else:
            return True, 2, solution_coords[1:]
    else:
        return True, 0, solution_coords


def connect_map(path, endp, map, maxtry=200):
    global trycnt
    msize = map.shape[0]
    crtp = path[-1]
    pdir = np.sign(endp - crtp)
    moves = []
    last_move = []
    if pdir[0] != 0:
        moves.append(np.array([pdir[0], 0]))
        last_move.append(np.array([-pdir[0], 0]))
    else:
        last_move += [np.array([1, 0]), np.array([-1, 0])]

    if pdir[1] != 0:
        moves.append(np.array([0, pdir[1]]))
        last_move.append(np.array([0, -pdir[1]]))
    else:
        last_move += [np.array([0, 1]), np.array([0, -1])]

    moves = np.stack(moves + last_move)

    next_ps = (crtp + moves).astype(np.int)
    vd = np.all(next_ps >= 1, axis=1) & np.all(next_ps < (msize - 1), axis=1)
    for itt, nextp in enumerate(next_ps[vd]):
        trycnt += 1

        if trycnt > maxtry:
            return True, []

        if np.all(nextp == endp):
            return True, path + [nextp]

        if map[nextp[0], nextp[1]] == 0:
            map[nextp[0], nextp[1]] = 3
            rt, npath = connect_map(path + [nextp], endp, map)
            if rt:
                return rt, npath
            map[nextp[0], nextp[1]] = 0
    return False, path


def get_solution(obs, goal_select):
    obs_size = obs["image"].shape[0] // 2 + 3

    map = np.zeros((obs_size, obs_size))
    for vobj, objpos in zip(obs["available_obj"], obs["obj_pos"]):
        if vobj:
            map[objpos[0], objpos[1]] = 1

    agx, agy, agd = obs["agent"]
    map[agx, agy] = 2

    map[goal_select[0], goal_select[1]] = -1

    global trycnt
    trycnt = 0
    rt, cpath = connect_map([np.array([agx, agy])], goal_select, map)

    return rt, cpath

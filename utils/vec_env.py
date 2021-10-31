from multiprocessing import Process, Pipe, Queue
import multiprocessing
import gym
from argparse import Namespace
from typing import List, Any
import numpy as np


def get_minigrid_envs(full_args: Namespace, env_wrapper: Any, no_envs: int,
                      env_seed_offset: int = 10000):
    """ Minigrid action 6 is Done - useless"""
    envs = []
    args = full_args.main
    actual_procs = args.actual_procs

    max_actions = getattr(full_args.env_cfg, "max_actions", 6)

    env_args = getattr(full_args.env_cfg, "env_args", None)
    env_args = dict() if env_args is None else vars(env_args)

    env = gym.make(args.env, **env_args)
    env.action_space.n = max_actions
    env.max_steps = full_args.env_cfg.max_episode_steps
    env.unwrapped._env_proc_id = 0
    env = env_wrapper(env)
    env.seed(args.seed + env_seed_offset * 0)

    envs.append([env])
    chunk_size = int(np.ceil((no_envs - 1) / float(actual_procs)))
    if chunk_size > 0:
        for env_i in range(1, no_envs, chunk_size):
            env_chunk = []
            for i in range(env_i, min(env_i + chunk_size, no_envs)):
                env = gym.make(args.env, **env_args)
                env.action_space.n = max_actions
                env.max_steps = full_args.env_cfg.max_episode_steps
                env.unwrapped._env_proc_id = i
                env = env_wrapper(env)
                env.seed(args.seed + env_seed_offset * i)

                env_chunk.append(env)
            envs.append(env_chunk)

    return envs, chunk_size


def worker_multi(conn, conn_send, envs):
    envs = list(envs)

    while True:
        cmd, datas = conn.recv()
        if cmd == "step":
            for (env_idx, env), data in zip(envs, datas):
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                conn_send.put((env_idx, (obs, reward, done, info)))
        elif cmd == "reset":
            for env_idx, env in envs:
                obs = env.reset()
                conn_send.put((env_idx, obs))
        elif cmd == "exit":
            return
        else:
            raise NotImplementedError


def worker_multi_with_last_obs(conn, conn_send, envs):
    envs = list(envs)

    while True:
        cmd, datas = conn.recv()
        if cmd == "step":
            for (env_idx, env), data in zip(envs, datas):
                obs, reward, done, info = env.step(data)
                if done:
                    if info is None:
                        info = dict()
                    info["last_obs"] = obs
                    obs = env.reset()
                conn_send.put((env_idx, (obs, reward, done, info)))
        elif cmd == "reset":
            for env_idx, env in envs:
                obs = env.reset()
                conn_send.put((env_idx, obs))
        elif cmd == "exit":
            return
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """
        A concurrent execution of environments in multiple processes.
        Distribute envs to limited number of processes
    """

    def __init__(self, envs, worker_target=worker_multi):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.first_env = self.envs[0][0]
        self.observation_space = self.first_env.observation_space
        self.action_space = self.first_env.action_space

        self.locals = []

        self.no_envs = sum(map(len, self.envs[1:]))
        self.num_procs = self.no_envs + 1
        self.local_recv = remote_send = Queue()

        env_idx = 1
        self.env_idxs = []
        self.processes = []
        for env_b in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker_target, args=(remote, remote_send, zip(range(env_idx, env_idx+len(env_b)), env_b)))
            p.daemon = True
            p.start()
            remote.close()
            self.env_idxs.append([env_idx, env_idx+len(env_b)])
            env_idx += len(env_b)
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))

        results = [self.first_env.reset()] + [None] * self.no_envs
        no_recv = 0
        max_recv = self.no_envs

        local = self.local_recv
        while no_recv < max_recv:
            env_idx, r = local.get()
            results[env_idx] = r
            no_recv += 1
        return results

    def step(self, actions):
        # Send Chunck actions
        for local, action_idxs in zip(self.locals, self.env_idxs):
            local.send(("step", actions[action_idxs[0]:action_idxs[1]]))
        obs, reward, done, info = self.first_env.step(actions[0])
        if done:
            obs = self.first_env.reset()

        results = [(obs, reward, done, info)] + [None] * self.no_envs
        no_recv = 0
        max_recv = self.no_envs
        local = self.local_recv

        while no_recv < max_recv:
            env_idx, r = local.get()
            results[env_idx] = r
            no_recv += 1

        results = zip(*results)
        return results

    def render(self):
        raise NotImplementedError

    def close_procs(self):
        for local in self.locals:
            local.send(("exit", None))

        for p in self.processes:
            p.terminate()
            p.join()


class ParallelEnvWithLastObs(ParallelEnv):
    """
        Send last observation before reset.
    """

    def __init__(self, *args, **kwargs):
        multiprocessing.connection.BUFSIZE *= 10
        super().__init__(*args, worker_target=worker_multi_with_last_obs, **kwargs)

    def step(self, actions):
        # Send Chunck actions
        for local, action_idxs in zip(self.locals, self.env_idxs):
            local.send(("step", actions[action_idxs[0]:action_idxs[1]]))
        obs, reward, done, info = self.first_env.step(actions[0])
        if done:
            if info is None:
                info = dict()
            info["last_obs"] = obs
            obs = self.first_env.reset()

        results = [(obs, reward, done, info)] + [None] * self.no_envs
        no_recv = 0
        max_recv = self.no_envs
        local = self.local_recv

        while no_recv < max_recv:
            env_idx, r = local.get()
            results[env_idx] = r
            no_recv += 1

        results = zip(*results)
        return results


class ParallelEnvWithLastObsIndex(ParallelEnv):
    """
        Send last observation before reset.
    """

    def __init__(self, envs, **kwargs):
        all_envs = []
        for x in envs:
            all_envs += x
        all_envs = [[x] for x in all_envs]
        super().__init__(all_envs, worker_target=worker_multi_with_last_obs, **kwargs)

    def step(self, actions, idxs=None):
        if idxs is None:
            idxs = list(range(self.no_envs + 1))

        # Send Chunck actions
        locals = self.locals
        send_to_first = False
        results = [None] * (self.no_envs + 1)
        max_recv = len(idxs)

        for send_act, send_idx in zip(actions, idxs):
            if send_idx != 0:
                locals[send_idx-1].send(("step", [send_act]))
            else:
                send_to_first = True

        if send_to_first:
            obs, reward, done, info = self.first_env.step(actions[0])

            if done:
                if info is None:
                    info = dict()
                info["last_obs"] = obs
                obs = self.first_env.reset()

            results[0] = (obs, reward, done, info)
            max_recv -= 1

        no_recv = 0
        local = self.local_recv

        while no_recv < max_recv:
            env_idx, r = local.get()
            results[env_idx] = r
            no_recv += 1

        results = zip(*[x for x in results if x is not None])
        return results

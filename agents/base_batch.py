# Credits (inspired & adapted from) @ https://github.com/lcswillems/torch-rl

import torch
from typing import Tuple
import numpy as np

from utils.dictlist import DictList
from .base import AgentBase


class SimulateDict:
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, item):
        return self._obj


class BatchBase(AgentBase):
    """ The base class for the RL algorithms. """

    def __init__(self, *args, **kwargs):
        """
        cfg Should contain the following (at least for the base):
            :cfg num_frames_per_proc: int -
                the number of frames collected by every process for an update
        """
        super().__init__(*args, **kwargs)

        cfg = self.cfg

        # -- Store helpers values
        self.num_frames_per_proc = cfg.num_frames_per_proc
        self.use_gae_return = getattr(cfg, "use_gae_return", True)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        self._protect_env_data = True

        # Get from base
        recurrent = self.recurrent
        num_procs = self.num_procs
        device = self.device

        # -- Initialize experience data holders
        shape = (self.num_frames_per_proc, num_procs)

        if recurrent:
            assert self.num_frames_per_proc % self.recurrence == 0, "Use all observations!"

            # Get memory size from model
            self.memory = torch.zeros(shape[1], self.model.memory_size, device=device)
            self.memories = torch.zeros(*shape, self.model.memory_size, device=device)

        self.obs = self.env.reset()

        self.obss = [None] * (shape[0])
        self.mask = torch.zeros(shape[1], device=device)
        self.masks = torch.zeros(*shape, device=device)
        self.actions = torch.zeros(*shape, device=device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=device)
        self.rewards = torch.zeros(*shape, device=device)
        self.advantages = torch.zeros(*shape, device=device)
        self.log_probs = torch.zeros(*shape, device=device)
        self.termination_target = torch.zeros(*shape, device=device)
        self.op_used = torch.zeros(*shape, device=device)
        self.new_op_mask = torch.zeros(*shape, device=device)
        self.obs_termination_target = torch.zeros(num_procs, device=device)

        if not self.use_gae_return:
            self.rreturn = torch.zeros(*shape, device=device)

        self.log_episode_return = torch.zeros(num_procs, device=device)
        self.log_episode_num_frames = torch.zeros(num_procs, device=device)

        # -- Init evaluation environments and storage data
        if self.has_evaluator:
            eval_envs = self.eval_envs
            obs = eval_envs.reset()

            if self.recurrent:
                self.eval_memory = torch.zeros(len(obs), self.model.memory_size, device=device)

            self.eval_mask = torch.ones(len(obs), device=device)
            self.eval_rewards = torch.zeros(len(obs), device=device)

    def update_parameters(self) -> dict:
        """  [REPLACE] Implement agent training  """

        # -- Collect experiences
        exps, logs = self._collect_experiences()

        # -- Train Loop
        # ...

        return logs

    def _collect_experiences(self) -> Tuple[DictList, dict]:
        """ Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.
        * considers model with recurrent data

        Returns
        -------
        exps : DictList --> shapes: (num_frames_per_proc * num_envs, ...)
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward etc.
        """
        model = self.model
        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc
        preprocess_obss = self.preprocess_obss
        device = self.device
        recurrent = self.recurrent
        log_episode_return = self.log_episode_return
        log_episode_num_frames = self.log_episode_num_frames
        discount = self.discount
        gae_lambda = self.gae_lambda
        masks = self.masks
        values = self.values
        advantages = self.advantages
        rewards = self.rewards
        actions = self.actions
        obss = self.obss
        log_probs = self.log_probs
        termination_target = self.termination_target
        use_gae_return = self.use_gae_return
        protect_env_data = self._protect_env_data

        memory = None
        dtype = torch.float
        self.log_done_counter = 0
        self.log_return = log_return = []
        self.log_num_frames = log_num_frames = []

        for i in range(num_frames_per_proc):
            # -- Do one agent-environment interaction

            preprocessed_obs = preprocess_obss(self.obs, device=device)
            with torch.no_grad():
                if recurrent:
                    res_m = model(
                        preprocessed_obs, mask=self.mask,
                        memory=self.memory * self.mask.unsqueeze(1)
                    )
                else:
                    res_m = model(preprocessed_obs, mask=self.mask)

                action, act_log_prob, value = res_m["actions"], \
                                              res_m["act_log_probs"], \
                                              res_m["values"]

            obs, reward, done, info = self.env.step(self._process_actions_for_step(action, res_m))
            if not protect_env_data:
                obs, reward, done, info = list(obs), list(reward), list(done), list(info)

            # -- Process other useful information each step
            self._collect_experiences_step(i, obs, reward, done, info, preprocessed_obs, res_m)

            # -- Update experiences values
            obss[i] = self.obs
            termination_target[i] = self.obs_termination_target

            if "termination_target" in res_m:
                self.obs_termination_target = res_m["termination_target"]
                self.op_used[i] = res_m["crt_op_idx"]
                self.new_op_mask[i] = res_m["new_op_mask"]

            self.obs = obs

            if recurrent:
                self.memories[i] = self.memory
                self.memory = res_m["memory"]

            masks[i] = self.mask
            self.mask = torch.tensor(1.) - torch.tensor(done, device=device, dtype=dtype)

            actions[i] = action
            values[i] = value
            rewards[i] = torch.tensor(reward, device=device)
            log_probs[i] = act_log_prob

            # Update log values
            log_episode_return += torch.tensor(reward, device=device, dtype=dtype)
            log_episode_num_frames += torch.ones(num_procs, device=device)

            for j, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    log_return.append(log_episode_return[j].item())
                    log_num_frames.append(log_episode_num_frames[j].item())

            log_episode_return *= self.mask
            log_episode_num_frames *= self.mask

        # -- Add advantage and return to experiences
        preprocessed_obs = preprocess_obss(self.obs, device=device)
        with torch.no_grad():
            if recurrent:
                res_last_obs = model(preprocessed_obs, memory=self.memory * self.mask.unsqueeze(1))
                next_value = res_last_obs["values"]
            else:
                res_last_obs = model(preprocessed_obs)
                next_value = res_last_obs["values"]

        if not use_gae_return:
            rreturn = self.rreturn

        for i in reversed(range(num_frames_per_proc)):
            next_mask = masks[i + 1] if i < num_frames_per_proc - 1 else self.mask
            next_value = values[i + 1] if i < num_frames_per_proc - 1 else next_value
            next_advantage = advantages[i + 1] if i < num_frames_per_proc - 1 else 0

            delta = rewards[i] + discount * next_value * next_mask - values[i]
            advantages[i] = delta + discount * gae_lambda * next_advantage * next_mask

            if not use_gae_return:
                next_rreturn = rreturn[i + 1] if i < num_frames_per_proc - 1 else next_value
                rreturn[i] = next_rreturn * discount * next_mask + rewards[i]

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [obss[i][j]
                    for j in range(num_procs)
                    for i in range(num_frames_per_proc)]
        if recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = actions.transpose(0, 1).reshape(-1)
        exps.value = values.transpose(0, 1).reshape(-1)
        exps.reward = rewards.transpose(0, 1).reshape(-1)
        exps.advantage = advantages.transpose(0, 1).reshape(-1)
        exps.returnn = (exps.value + exps.advantage) if use_gae_return else \
            self.rreturn.transpose(0, 1).reshape(-1)
        exps.log_prob = log_probs.transpose(0, 1).reshape(-1)
        exps.termination_target = termination_target.transpose(0, 1).reshape(-1)

        # Pre-process experiences
        exps.obs = preprocess_obss(exps.obs, device=device)
        exps.res_last_obs = SimulateDict(res_last_obs)

        log = {
            "return_per_episode": log_return,
            "num_frames_per_episode": log_num_frames,
            "num_frames": [self.num_frames]
        }

        extra_logs = self._collect_experiences_finished()
        log.update(extra_logs)

        return exps, log

    def _process_actions_for_step(self, actions, model_results):
        return actions.cpu().numpy()

    def _get_batches_starting_indexes(self, batch_size: int, shift_start_idx: bool = False):
            """
                Get batches of indexes. Take recurrence into consideration to separate indexes.

            """
            recurrence = self.recurrence
            num_frames = self.num_frames

            indexes = np.arange(0, num_frames, recurrence)
            indexes = np.random.permutation(indexes)

            # Shift starting indexes by recurrence//2
            if shift_start_idx:
                # Eliminate last index from environment trajectory (so not to overshoot)
                indexes = indexes[(indexes + recurrence) % self.num_frames_per_proc != 0]
                indexes += recurrence // 2

            num_indexes = batch_size // recurrence
            batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes),
                                                                                num_indexes)]
            return batches_starting_indexes

    def _save_experience(self, update_step: int, exps: DictList, logs: dict):
        if self.save_experience <= 0:
            return

        # TODO Hackish way to save experience -
        norm_value = self.max_image_value

        nstep = self.save_experience * self.num_frames_per_proc
        experience = dict()
        experience["logs"] = logs
        experience["obs_image"] = (exps.obs.image[:nstep].cpu() * norm_value).byte()
        experience["mask"] = exps.mask[:nstep].cpu()
        experience["action"] = exps.action[:nstep].cpu()
        experience["reward"] = exps.reward[:nstep].cpu()
        experience["num_procs"] = self.save_experience
        experience["frames_per_proc"] = self.num_frames_per_proc
        experience["norm_value"] = norm_value
        torch.save(experience, f"{self.experience_dir}/exp_update_{update_step}")



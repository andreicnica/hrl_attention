# Credits (inspired & adapted from) @ https://github.com/lcswillems/torch-rl

import numpy as np
import torch
from typing import Tuple
from utils.dictlist import DictList

from agents.base_batch import SimulateDict
from agents.ppo import PPO
from utils.logging_utils import LogCfg


BASE_LOGS = [
    LogCfg("entropy", True, "l", "e:u,", ":.2f", False, True),
    LogCfg("value", True, "l", "v:u", ":.2f", False, False),
    LogCfg("policy_loss", True, "l", "pL:u", ":.2f", False, False),
    LogCfg("value_loss", True, "l", "vL:u", ":.2f", False, False),
    LogCfg("grad_norm", True, "l", "g:u", ":.2f", False, False)
]

EVAL_LOG = [LogCfg("eval_r", True, "a", "e:u", ":.2f", False, False)]


class PPOSmdp(PPO):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, *args, **kwargs):
        """
        cfg Should contain the following (at least for the base):
            :cfg num_frames_per_proc: int -
                the number of frames collected by every process for an update
        """
        self.memory = torch.zeros(1)  # Just to have in case of no recurrence

        super().__init__(*args, **kwargs)

        self.train_smdp_only = getattr(self.cfg, "train_smdp_only", True)

        shape = (self.num_frames_per_proc, self.num_procs)

        self.extra_obs = [list() for _ in range(self.num_procs)]
        self.extra_obss = [None] * (shape[0])
        self.num_steps = torch.zeros(*shape, device=self.device)
        self.log_task_success = list()

        self._logs_train_config += [
            LogCfg("task_success", True, "a", "TrR:u,", ":.2f", True, True)
        ]

    def update_parameters(self):
        self._update_step += 1

        # -- Collect experiences
        exps, logs = self._collect_experiences()

        if self.save_experience > 0:
            self._save_experience(self._update_step, exps, logs)

        # -- Training config variables
        model = self.model
        optimizer = self.optimizer
        batch_size = self.batch_size
        recurrence = self.recurrence
        recurrent = self.recurrent
        clip_eps = self.clip_eps
        entropy_coef = self.entropy_coef
        value_loss_coef = self.value_loss_coef
        max_grad_norm = self.max_grad_norm
        memory = None  # type: torch.Tensor

        # -- Initialize log values
        log_entropies = []
        log_values = []
        log_policy_losses = []
        log_value_losses = []
        log_grad_norms = []

        for epoch_no in range(self.epochs):
            for obs_inds in self._custom_get_batches_starting_indexes(len(exps.obs), batch_size):

                inds = exps.obs_ids[obs_inds]

                # -- Initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # -- Initialize memory
                if model.recurrent:
                    memory = exps.memory[inds]

                for i in range(recurrence):
                    # -- Create a sub-batch of experience
                    sb = exps[inds + i]
                    obs = exps.obs[obs_inds + i]

                    # -- Compute loss
                    if recurrent:
                        res_m = model(obs, no_interest=True, memory=memory * sb.mask)
                        dist, value, memory = res_m["dist"], res_m["values"], res_m["memory"]

                    else:
                        res_m = model(obs, no_interest=True)
                        dist, value = res_m["dist"], res_m["values"]

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -clip_eps, clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - entropy_coef * entropy + value_loss_coef * value_loss

                    # -- Update batch values
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # -- Update memories for next epoch
                    if recurrent and i < recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # -- Update batch values
                batch_entropy /= recurrence
                batch_value /= recurrence
                batch_policy_loss /= recurrence
                batch_value_loss /= recurrence
                batch_loss /= recurrence

                # -- Update actor-critic
                optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                # -- Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # -- Log some values
        logs["entropy"] = [np.mean(log_entropies)]
        logs["value"] = [np.mean(log_values)]
        logs["policy_loss"] = [np.mean(log_policy_losses)]
        logs["value_loss"] = [np.mean(log_value_losses)]
        logs["grad_norm"] = [np.mean(log_grad_norms)]

        return logs

    def _custom_get_batches_starting_indexes(self, num_elem, batch_size: int,
                                             shift_start_idx: bool = False):
            """
                Get batches of indexes. Take recurrence into consideration to separate indexes.

            """
            assert self.recurrence == 1, "Not implemented for recurrent network"
            recurrence = 1
            indexes = np.arange(0, num_elem, recurrence)
            indexes = np.random.permutation(indexes)

            num_indexes = batch_size // recurrence
            batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes),
                                                                                num_indexes)]
            return batches_starting_indexes

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
        num_steps = self.num_steps

        memory = None
        dtype = torch.float
        self.log_done_counter = 0
        self.log_return = log_return = []
        self.log_num_frames = log_num_frames = []
        self.log_task_success = log_task_success = list()

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

            num_steps[i] = preprocessed_obs.num_steps
            self.extra_obss[i] = [x["extra_obs"][:-1] if len(x["extra_obs"]) > 1 else [] for x in info]
            self.obs = obs
            # Save trajectory without next state

            if recurrent:
                self.memories[i] = self.memory
                self.memory = res_m["memory"]

            masks[i] = self.mask
            self.mask = torch.tensor(1.) - torch.tensor(done, device=device, dtype=dtype)

            actions[i] = action
            log_probs[i] = act_log_prob
            values[i] = value
            rewards[i] = torch.tensor(reward, device=device)
            if "termination_target" in res_m:
                termination_target[i] = res_m["termination_target"]

            # Update log values
            log_episode_return += torch.tensor(reward, device=device, dtype=dtype)
            log_episode_num_frames += torch.ones(num_procs, device=device)

            for j, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    log_return.append(log_episode_return[j].item())
                    log_num_frames.append(log_episode_num_frames[j].item())
                    log_task_success.append(int(info[j].get("full_task_achieved", 0)))

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

            num_step = num_steps[i + 1] if i < num_frames_per_proc - 1 else \
                preprocessed_obs.num_steps

            delta = rewards[i] + (discount ** num_step) * next_value * next_mask - values[i]
            advantages[i] = delta + (discount ** num_step) * gae_lambda * next_advantage * next_mask

            if not use_gae_return:
                next_rreturn = rreturn[i + 1] if i < num_frames_per_proc - 1 else next_value
                rreturn[i] = next_rreturn * (discount ** num_step) * next_mask + rewards[i]

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

        obs_ids = list(range(len(exps.obs)))

        if not self.train_smdp_only:
            extra_obs_l = [self.extra_obss[i][j]
                           for j in range(num_procs)
                           for i in range(num_frames_per_proc)]

            extra_obs = []
            extra_obs_ids = []
            for qi in obs_ids:
                extra_obs += extra_obs_l[qi]
                extra_obs_ids += [qi] * len(extra_obs_l[qi])

            obs_ids += extra_obs_ids
            exps.obs += extra_obs

        exps.obs_ids = np.array(obs_ids)

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

        # torch.save(dict(exps), f"results/exps/exps_{self._update_step}")

        # We do not need aff or interest for training policy over options
        for ooo in exps.obs:
            if "interest" in ooo:
                ooo.pop("interest")
                ooo.pop("op_aff")

        # Pre-process experiences
        exps.obs = preprocess_obss(exps.obs, device=device)
        exps.res_last_obs = SimulateDict(res_last_obs)

        log = {
            "return_per_episode": log_return,
            "task_success": log_task_success,
            "num_frames_per_episode": log_num_frames,
            "num_frames": [self.num_frames]
        }

        extra_logs = self._collect_experiences_finished()
        log.update(extra_logs)

        return exps, log

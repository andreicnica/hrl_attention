# Credits (inspired & adapted from) @ https://github.com/lcswillems/torch-rl

import numpy as np
import torch

from agents.base_batch import BatchBase
from utils.logging_utils import LogCfg


BASE_LOGS = [
    LogCfg("entropy", True, "l", "e:u,", ":.2f", False, True),
    LogCfg("value", True, "l", "v:u", ":.2f", False, False),
    LogCfg("policy_loss", True, "l", "pL:u", ":.2f", False, False),
    LogCfg("value_loss", True, "l", "vL:u", ":.2f", False, False),
    LogCfg("grad_norm", True, "l", "g:u", ":.2f", False, False),
]

EVAL_LOG = [LogCfg("eval_r", True, "a", "e:u", ":.2f", False, False)]


class PPO(BatchBase):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cfg

        # -- Load configs specific to this agent
        self.entropy_coef = getattr(cfg, "entropy_coef", 0.01)
        self.value_loss_coef = getattr(cfg, "value_loss_coef", 0.5)
        self.max_grad_norm = getattr(cfg, "max_grad_norm", 0.5)
        self.clip_eps = getattr(cfg, "clip_eps", 0.)
        self.epochs = getattr(cfg, "epochs", 4)
        self.batch_size = getattr(cfg, "batch_size", 256)

        optimizer = getattr(cfg, "optimizer", "Adam")
        optimizer_args = getattr(cfg, "optimizer_args", {})
        optimizer_args = vars(optimizer_args)
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), **optimizer_args)

        assert self.batch_size % self.recurrence == 0, "Use all observations!"

        self._logs_train_config += BASE_LOGS
        self._logs_eval_config += EVAL_LOG

        self._update_step = 0

    def get_checkpoint(self) -> dict:
        return dict({"optimizer": self.optimizer.state_dict()})

    def load_checkpoint(self, agent_data: dict):
        if "optimizer" in agent_data:
            self.optimizer.load_state_dict(agent_data["optimizer"])

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
            for inds in self._get_batches_starting_indexes(batch_size, shift_start_idx=False):

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

                    # -- Compute loss
                    if recurrent:
                        res_m = model(sb.obs, memory=memory * sb.mask)
                        dist, value, memory = res_m["dist"], res_m["values"], res_m["memory"]

                    else:
                        res_m = model(sb.obs)
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

    def evaluate(self, eval_key=None):
        # -- Evaluation config variables
        env = self.eval_envs
        eval_episodes = self.eval_episodes
        preprocess_obss = self.preprocess_obss
        device = self.device
        recurrent = self.recurrent
        model = self.model
        rewards = self.eval_rewards
        mask = self.eval_mask.fill_(1).unsqueeze(1)

        memory = None
        obs = env.reset()
        if recurrent:
            memory = self.eval_memory
            memory.zero_()

        # -- Initialize log values
        num_envs_results = 0
        log_reshaped_return = []
        rewards.zero_()
        log_episode_reshaped_return = torch.zeros_like(rewards)

        # Wait for eval_episodes to finish
        while num_envs_results < eval_episodes:

            # -- Run eva environment steps
            preprocessed_obs = preprocess_obss(obs, device=device)
            with torch.no_grad():
                if recurrent:
                    dist, value, memory = model(preprocessed_obs, memory=memory * mask)
                else:
                    dist, value = model(preprocessed_obs)
                action = dist.sample()

            next_obs, reward, done, info = env.step(action.cpu().numpy())

            mask = (torch.tensor(1.) - torch.tensor(done, device=device, dtype=torch.float))
            mask.unsqueeze_(1)

            rewards = torch.tensor(reward, device=self.device)
            log_episode_reshaped_return += rewards

            # -- Log evaluation reward from finished episodes
            for j, done_ in enumerate(done):
                if done_:
                    num_envs_results += 1
                    if eval_key is None:
                        log_reshaped_return.append(log_episode_reshaped_return[j].item())
                    else:
                        log_reshaped_return.append(info[j][eval_key])

            log_episode_reshaped_return *= mask.squeeze(1)
            obs = next_obs

        return {"eval_r": [np.mean(log_reshaped_return[:eval_episodes])]}

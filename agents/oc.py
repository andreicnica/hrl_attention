# Credits (inspired & adapted from) @ https://github.com/lcswillems/torch-rl

import numpy as np
import torch
from typing import List

from utils.dictlist import DictList
from agents.base_batch import BatchBase
from utils.logging_utils import LogCfg


BASE_LOGS = [
    LogCfg("entropy", True, "a", "e:u,", ":.2f", False, True),
    LogCfg("value", True, "a", ",v:u", ":.2f", False, False),
    LogCfg("policy_loss", True, "a", "pL:u", ":.2f", False, False),
    LogCfg("value_loss", True, "a", "vL:u", ":.2f", False, False),
    LogCfg("grad_norm", True, "a", "g:u", ":.2f", False, False),
]

EVAL_LOG = [LogCfg("eval_r", True, "a", "e:u", ":.2f", False, False)]


class OptionCritic(BatchBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cfg

        # -- Load configs specific to this agent
        self.entropy_coef = getattr(cfg, "entropy_coef", 0.01)
        self.value_loss_coef = getattr(cfg, "value_loss_coef", 0.5)
        self.max_grad_norm = getattr(cfg, "max_grad_norm", 0.5)

        self.eps_greedy = getattr(cfg, "eps_greedy", 0.1)
        self.beta_reg = getattr(cfg, "beta_reg", 0.01)
        self.beta_loss_coef = getattr(cfg, "beta_loss_coef", 0.5)

        optimizer = getattr(cfg, "optimizer", "Adam")
        optimizer_args = getattr(cfg, "optimizer_args", {})
        optimizer_args = vars(optimizer_args)
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), **optimizer_args)

        self._logs_train_config += BASE_LOGS
        self._logs_eval_config += EVAL_LOG

        self._update_step = 0

        shape = (self.num_frames_per_proc, self.num_procs)
        self.beta_adv = torch.zeros(*shape, device=self.device)

        self.model_results = [None] * self.num_frames_per_proc
        self._collect_experiences_step(-1, self.obs, None, None, None, None, None)

    def get_checkpoint(self) -> dict:
        return dict({"optimizer": self.optimizer.state_dict()})

    def load_checkpoint(self, agent_data: dict):
        if "optimizer" in agent_data:
            self.optimizer.load_state_dict(agent_data["optimizer"])

    def _collect_experiences_step(self, frame_id: int, obs: List[DictList], reward: List[float],
                                  done: List[bool], info: List[dict], prev_obs: DictList,
                                  model_result: dict):

        prev_op = [0] * self.num_procs if model_result is None else \
            model_result["crt_op_idx"].cpu().numpy()

        for ix in range(len(obs)):
            obs[ix]["prev_option"] = prev_op[ix]

        if frame_id >= 0:
            self.model_results[frame_id] = model_result

    def update_parameters(self):
        self._update_step += 1

        # -- Collect experiences
        exps, logs = self._collect_experiences()

        # ==========================================================================================
        # -- Update exps with option advantages
        discount = self.discount

        res_last_obs = exps.res_last_obs[0]

        prev_op = res_last_obs["prev_op"]
        next_op_qvalue = res_last_obs["op_qvalues"].gather(1, prev_op.unsqueeze(1)).squeeze(1)
        next_max_op_qvalue = res_last_obs["op_qvalues"].max(dim=1).values
        next_terminations = res_last_obs["op_terminations"].gather(1, prev_op.unsqueeze(
            1)).squeeze(1)

        ret = (1 - next_terminations) * next_op_qvalue + next_terminations * next_max_op_qvalue

        masks = self.masks
        rewards = self.rewards
        values = self.values
        rreturn = self.rreturn
        advantages = self.advantages
        ret_m = self.model_results
        num_frames_per_proc = self.num_frames_per_proc
        beta_adv = self.beta_adv
        eps_g = self.eps_greedy
        beta_reg = self.beta_reg
        # ==========================================================================================

        for i in reversed(range(num_frames_per_proc)):
            next_mask = masks[i + 1] if i < num_frames_per_proc - 1 else self.mask
            op_qvalues = ret_m[i]["op_qvalues"]
            prev_op = ret_m[i]["prev_op"]

            # calculate option value and advantage value
            ret = rewards[i] + discount * next_mask * ret

            adv = ret - values[i]

            rreturn[i] = ret
            advantages[i] = adv

            v = op_qvalues.max(dim=1).values * (1 - eps_g) + op_qvalues.mean(dim=1) * eps_g
            q = op_qvalues.gather(1, prev_op.unsqueeze(1)).squeeze(1)
            beta_adv[i] = q - v + beta_reg

        # ==========================================================================================

        if self.save_experience > 0:
            self._save_experience(self._update_step, exps, logs)

        # -- Training config variables
        model = self.model
        optimizer = self.optimizer
        recurrence = self.recurrence
        recurrent = self.recurrent
        entropy_coef = self.entropy_coef
        value_loss_coef = self.value_loss_coef
        max_grad_norm = self.max_grad_norm
        beta_loss_coef = self.beta_loss_coef
        memory = None  # type: torch.Tensor
        batch_size = None

        # -- Initialize log values
        log_entropies = []
        log_values = []
        log_policy_losses = []
        log_value_losses = []
        log_grad_norms = []

        for inds in self._get_batches_starting_indexes(batch_size):
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

                op_terminate = res_m["op_terminate"]
                new_op_mask = res_m["new_op_mask"]

                entropy = dist.entropy().mean()

                policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

                value_loss = (value - sb.returnn).pow(2).mean()

                beta_loss = (op_terminate * beta_adv.detach() *
                             (1 - new_op_mask.float().detach())).mean()

                loss = policy_loss - entropy_coef * entropy + value_loss_coef * value_loss + \
                       beta_loss_coef * beta_loss

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
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None
            ) ** 0.5
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

    def _get_batches_starting_indexes(self, batch_size: int, shift_start_idx: bool = False):
        recurrence = self.recurrence
        indexes = np.arange(0, self.num_frames, self.recurrence)
        num_indexes = len(indexes) if batch_size is None else batch_size // recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes),
                                                                              num_indexes)]

        return batches_starting_indexes


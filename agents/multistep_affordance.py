from typing import List
import logging
import torch
import numpy as np
import torch.nn.functional as F

from utils.dictlist import DictList
from utils.logging_utils import LogCfg
from agents.base_batch_and_trajectory import TrajectoryBufferWithBatchBase
from utils import get_intent_completion_method
from utils.data_primitives import Experience
import copy


EVAL_LOG = [LogCfg("eval_r", True, "l", "e:u", ":.2f", True, True)]


logger = logging.getLogger(__name__)


def append_to_logs(logs: dict, new_log: dict):
    for k, v in new_log.items():
        if k in logs:
            logs[k] += v
        else:
            logs[k] = v


class MultistepAffordances(TrajectoryBufferWithBatchBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cfg

        # -- Load configs specific to this agent
        self.intent_size = cfg.intent_size
        self.batch_trajectories = cfg.batch_trajectories
        self.use_affordance_to_mask_model = cfg.use_affordance_to_mask_model
        self.affordance_mask_threshold = cfg.affordance_mask_threshold
        self.train_options = cfg.train_options
        self.use_model = cfg.use_model
        self.intent_max_horizon = cfg.intent_max_horizon
        self.intent_achieved_th = cfg.intent_achieved_th
        self.affordance_discount = cfg.affordance_discount
        self.affordance_mask_used_op = cfg.affordance_mask_used_op
        self.use_affordances = cfg.use_affordances
        self.affordance_train_epochs = cfg.affordance_train_epochs
        self.aff_loss_balance = getattr(cfg, "aff_loss_balance", False)

        aff_criterion_cfg = cfg.aff_criterion
        self.aff_criterion_type = aff_criterion_cfg.aff_criterion_type
        if self.aff_criterion_type == "reinforce":
            self.reinforce_loss_coeff = getattr(aff_criterion_cfg, "reinforce_loss_coeff", 0.05)
        elif self.aff_criterion_type == "mse":
            self.norm_aff_target = getattr(aff_criterion_cfg, "norm_aff_target", 3)

        self.clip_eps = getattr(cfg, "clip_eps", 0.)
        self.epochs = getattr(cfg, "epochs", 4)
        self.batch_size = getattr(cfg, "batch_size", 256)
        self.aff_batch_size = getattr(cfg, "aff_batch_size", 0)
        self.aff_use_intermediary_ic = getattr(cfg, "aff_use_intermediary_ic", False)
        self.ic_reward_each_step = getattr(cfg, "ic_reward_each_step", True)
        assert self.ic_reward_each_step, "Not implemented without reward each step"

        self.ic_reward_negative = getattr(cfg, "ic_reward_negative", False)
        self.send_intent_to_env = getattr(cfg, "send_intent_to_env", False)
        self._reset_memory_on_new_op = getattr(cfg, "reset_memory_on_new_op", False) \
                                       and self.recurrent
        self._reset_memory_on_ic = getattr(cfg, "reset_memory_on_ic", False)
        self._next_mask = None

        self.ic_m = get_intent_completion_method(cfg.intent_completion_method, cfg)

        # So we can adjust reward on _collect_experiences_step
        self._protect_env_data = False

        # -- Optimizer transition model
        if self.cfg.use_model:
            self.optimizer_tr_model = None
            raise NotImplementedError

        # -- Optimizer for aff model
        optimizer = getattr(cfg, "optimizer_aff", "SGD")
        optimizer_args = getattr(cfg, "optimizer_args_aff", {})
        optimizer_args = vars(optimizer_args)
        self.optimizer_aff = getattr(torch.optim, optimizer)(
            self.model.affordance_network.parameters(), **optimizer_args
        )

        # ==========================================================================================
        # -- Option configs
        optimizer = getattr(cfg, "optimizer_op", "SGD")
        optimizer_args = getattr(cfg, "optimizer_args_op", {})
        optimizer_args = vars(optimizer_args)
        self.optimizer_op = getattr(torch.optim, optimizer)(
            self.model.options.parameters(), **optimizer_args
        )

        self._update_step = 0
        self.entropy_coef = getattr(cfg, "entropy_coef", 0.01)
        self.value_loss_coef = getattr(cfg, "value_loss_coef", 0.5)
        self.termination_loss_coef = getattr(cfg, "termination_loss_coef", 0.0)
        self.max_grad_norm = getattr(cfg, "max_grad_norm", 0.5)

        self.model_results = [None] * self.num_frames_per_proc
        self._collect_experiences_step(-1, self.obs, None, None, None, None, None)

        # ==========================================================================================
        # Add log for update_step: to print & write to csv

        self._logs_train_config += [
            LogCfg("entropy", False, "l", "e:u,", ":.2f", False, True),
            LogCfg("value", False, "l", ",v:u", ":.2f", False, False),
            LogCfg("policy_loss", False, "l", "pL:u", ":.2f", False, False),
            LogCfg("value_loss", False, "l", "vL:u", ":.2f", False, False),
            LogCfg("termination_loss", False, "l", "tL:u", ":.2f", False, False),
            LogCfg("termination_p_loss", False, "l", "tPL:u", ":.2f", False, False),
            LogCfg("termination_n_loss", False, "l", "tNL:u", ":.2f", False, False),
            LogCfg("grad_norm", False, "l", "g:u", ":.2f", False, False),
        ]

        self._logs_train_config += [
            LogCfg("trajectories_cnt", False, "l", "t", "", True, False),
            LogCfg("aff_loss", False, "l", "aff_loss", ":.4f", True, False),
            LogCfg("aff_loss_aff", False, "l", "aff", ":.4f", True, False),
            LogCfg("aff_loss_NOaff", False, "l", "NOaff", ":.4f", True, False),
            LogCfg("model_loss", False, "l", "model_loss", ":.4f", False, False),
            LogCfg("model_batch_affordable", False, "l", "mAFF%", ":.4f", False, False),
            LogCfg("model_acc", False, "l", "model_acc", ":.4f", False, False),
            LogCfg("intent_completion", True, "a", "ic", ":.4f", True, True),
            LogCfg("cnt_intent_completion", False, "l", "cnt_ic", ":.4f", True, False),
        ]

        num_op = self.model.num_options
        self._logs_train_config += [
            LogCfg(f"o{i}", True, "a", f"o{i}", ":.4f", False, False) for i in range(num_op)
        ]
        self._logs_train_config += [
            LogCfg("oe", False, "l", "oe", ":.4f", False, False),
        ]

        self._logs_train_config += [
            LogCfg(f"oic{i}", False, "l", f"oic{i}", ":.4f", False, False) for i in range(num_op)
        ]
        self._logs_train_config += [
            LogCfg(f"oPic{i}", False, "l", f"oPic{i}", ":.4f", False, False) for i in range(num_op)
        ]
        self._logs_train_config += [
            LogCfg("ic_balanced", True, "l", "icB", ":.4f", True, True),
            LogCfg("Pic_balanced", True, "l", "PicB", ":.4f", True, True),
            LogCfg("ic_without_done", True, "l", "icND", ":.4f", True, True),
        ]

        self._logs_eval_config += EVAL_LOG

    def _train_step_affordances(
            self, trajectory: List[List[Experience]], trajectory_info: List[dict]
    ):
        """ Train affordance network.
        Args:
            trajectory: A list of experiences
            trajectory_info: List of information about experiences (e.g. intent_completion)
        """
        affordance_network = self.model.affordance_network
        affordance_optimizer = self.optimizer_aff
        get_batch = self._get_batch_affordances
        aff_criterion_type = self.aff_criterion_type
        aff_loss_balance = self.aff_loss_balance

        log_loss, log_loss_aff, log_loss_no_aff = 0, [], []
        bix = 0
        unshaped_preds = None
        mask_used_op = self.affordance_mask_used_op

        num_epochs = 5

        for ep in range(num_epochs):
            for bix, (s_t, op_id, intent_target) in enumerate(get_batch(trajectory, trajectory_info)):

                affordance_optimizer.zero_grad()

                preds = affordance_network(s_t, op_id)
                unshaped_preds = preds

                if mask_used_op:
                    mask = torch.zeros_like(intent_target)
                    mask.scatter_(1, op_id.unsqueeze(1).long(), 1)
                    intent_target *= mask

                    intent_target = intent_target.gather(1, op_id.unsqueeze(1).long())

                no_aff = intent_target.sum(dim=1) == 0
                loss, loss_aff, loss_no_aff = None, None, None

                cnt_non_aff = no_aff.sum().item() # TODO test
                cnt_aff = no_aff.numel() - cnt_non_aff
                aff_weights = torch.ones_like(intent_target)
                if aff_loss_balance and cnt_non_aff != 0 and cnt_aff != 0:
                    if cnt_aff > cnt_non_aff:
                        aff_weights[no_aff] *= cnt_aff / float(cnt_non_aff)
                    else:
                        aff_weights[~no_aff] *= cnt_non_aff / float(cnt_aff)

                if aff_criterion_type == "default":
                    criterion = F.binary_cross_entropy
                    loss = criterion(preds, intent_target, reduction="none", weight=aff_weights)
                    loss_aff = loss[~no_aff].detach()
                    loss_no_aff = loss[no_aff].detach()
                    loss = loss.mean()
                elif aff_criterion_type == "reinforce":
                    log_probs = torch.log(preds.gather(1, op_id.long().unsqueeze(1)) + 1e-10)
                    affordance_advantage = 2 * intent_target.gather(1, op_id.long().unsqueeze(1)) - 1
                    afford_loss = (-log_probs * affordance_advantage)
                    loss = afford_loss.mean()
                    loss_aff = torch.zeros(1)
                    loss_no_aff = torch.zeros(1)
                    loss += loss * self.reinforce_loss_coeff
                elif aff_criterion_type == "mse":
                    criterion = F.mse_loss

                    if self.norm_aff_target > 0.:
                        intent_target = intent_target * self.norm_aff_target - self.norm_aff_target / 2
                        intent_target.detach_()

                    loss = criterion(preds, intent_target, reduction="none")
                    loss_aff = loss[~no_aff].detach()
                    loss_no_aff = loss[no_aff].detach()

                    # Weigh to be equal no_aff and ~no_aff
                    if aff_loss_balance:
                        if cnt_non_aff != 0 and cnt_aff != 0:
                            loss[no_aff] *= cnt_aff
                            loss[~no_aff] *= cnt_non_aff
                            loss /= (cnt_aff * cnt_non_aff)

                    loss = loss.mean()

                loss.backward()  # Nan values for aff or non_aff
                affordance_optimizer.step()

                log_loss += loss.item()
                log_loss_aff.append(loss_aff)
                log_loss_no_aff.append(loss_no_aff)

        log_loss /= (bix + 1) * num_epochs
        log_loss_aff = torch.cat(log_loss_aff).mean().item()
        log_loss_no_aff = torch.cat(log_loss_no_aff).mean().item()
        return [log_loss, log_loss_aff, log_loss_no_aff], unshaped_preds

    def _get_batch_affordances(
            self, trajectory: List[List[Experience]], trajectory_info: List[dict]
    ):

        aff_batch_size = self.aff_batch_size
        intent_size = self.intent_size
        aff_discount = self.affordance_discount
        aff_use_intermediary_ic = self.aff_use_intermediary_ic

        s_t, op_id_t, completion = [], [], []

        for tr, tri in zip(trajectory, trajectory_info):  # loop over trajectories
            next_ic = None
            op_id = tri["option"]

            for ix in range(len(tr))[::-1]:  # loop over transitions
                if tr[ix].model_result is None:  # last obs before env reset
                    continue

                if next_ic is None:
                    # Discount intent completion signal for affordance
                    next_ic = tri["intent_completion"]
                elif aff_use_intermediary_ic and tr[ix].ic[op_id] != 0:
                    next_ic = tr[ix].ic
                else:
                    next_ic = aff_discount * next_ic
                tr[ix].discounted_intent = next_ic

                obs = torch.transpose(torch.transpose(tr[ix].obs.image, 0, 2), 1, 2).contiguous()
                s_t.append(obs)

                completion.append(next_ic)

                op_id_t.append(op_id)

        s_t = torch.stack(s_t, dim=0)
        op_id_t = torch.tensor(op_id_t, device=self.device).float()
        completion = torch.tensor(completion, device=self.device).float()
        completion = completion[:, :intent_size]

        if aff_batch_size == 0:
            yield s_t, op_id_t, completion
        else:
            indexes = torch.randperm(len(s_t))
            for i in range(0, len(indexes), aff_batch_size):
                sidx = indexes[i: i + aff_batch_size]
                yield s_t[sidx], op_id_t[sidx], completion[sidx]

    def _train_step_model(self, trajectory: List[Experience], affordances: torch.Tensor):
        raise NotImplementedError

    def update_parameters(self) -> dict:
        """  Implement agent training  """
        use_affordances = self.use_affordances
        use_model = self.use_model
        num_procs = self.num_procs
        num_frames_per_proc = self.num_frames_per_proc

        batch_trajectories = self.batch_trajectories

        logs = dict()

        # ===============================================
        # Train loop
        # ===============================================

        # Step 1: Collect data.
        # ----------------------

        env_steps = 0
        new_tr = []  # type: List[List[Experience]]
        new_tr_info = []  # type: List[dict]
        no_run = True

        while len(new_tr) < batch_trajectories or no_run:
            # Run num_frames_per_proc steps in num_procs x environments
            exps, tr, tr_info, log_collect = self._collect_experiences()

            new_tr += tr
            new_tr_info += tr_info
            env_steps += num_frames_per_proc
            append_to_logs(logs, log_collect)

            if self.train_options:
                op_train_logs = self._train_step_options(exps)
                append_to_logs(logs, op_train_logs)

            no_run = False

        logs["trajectories_cnt"] = [len(new_tr)]
        logs["num_frames"] = [env_steps * num_procs]

        # Step 2: Calculate intent completion for trajectories.
        #  new_tr is expected to contain the list of trajectories (list ot transitions)
        # ----------------------

        # Calculated in the _collect_experiences_step method & added to tr_info
        logs["intent_completion"] = [x["intent_completion"][x["option"]] for x in new_tr_info]

        ic_without_done = []
        for iii, x in enumerate(new_tr_info):
            if new_tr[iii][-1].model_result is not None:
                ic_without_done.append(logs["intent_completion"][iii])
        logs["ic_without_done"] = [np.mean(ic_without_done)]

        logs["cnt_intent_completion"] = [sum(logs["intent_completion"])]

        # -- Log options used
        num_options = self.model.num_options
        options_cnt = torch.zeros(num_options)
        options_available_cnt = torch.zeros(num_options)
        options_ic = torch.zeros(num_options)

        op_av_cnt = "available_obj" in tr[0][0].obs  # Only for some envs

        for io in range(len(options_cnt)):
            logs[f"o{io}"] = [0] * len(new_tr)
        for itr, x in enumerate(new_tr_info):
            opid = x["option"]
            options_cnt[opid] += 1
            if op_av_cnt and tr[itr][0].obs.available_obj[opid]:
                options_available_cnt[opid] += 1
            options_ic[opid] += x["intent_completion"][opid]
            logs[f"o{opid}"][itr] = 1

        options_p_ic = options_ic / (options_available_cnt + 0.0000001)
        options_ic /= options_cnt
        logs[f"ic_balanced"] = [options_ic.mean().item()]
        logs[f"Pic_balanced"] = [options_p_ic[options_available_cnt > 0].mean().item()]

        options_ic = torch.cat([options_ic, torch.zeros(num_options - len(options_ic))])
        options_p_ic = torch.cat([options_p_ic, torch.zeros(num_options - len(options_p_ic))])
        for io in range(num_options):
            logs[f"oic{io}"] = [options_ic[io].item()]
            logs[f"oPic{io}"] = [options_p_ic[io].item()]

        options_cnt /= options_cnt.sum()
        options_cnt = options_cnt[:self.intent_size] + 0.0000001  # add eps
        logs[f"oe"] = [(-(options_cnt * options_cnt.log()).sum()).item()]

        # Step 3: Train Affordance model
        # ----------------------
        if use_affordances and len(new_tr) > 0:
            aff_loss, affordance_predictions = self._train_step_affordances(new_tr, new_tr_info)
        else:
            aff_loss = [-1., 1., 1.]

        # Step 3: Train Probabilistic transition model
        # ----------------------

        # Train transition model.
        acc, ptrain = 0., 0.
        if use_model:
            raise NotImplementedError
            # model_loss, acc, num_elem = self._train_step_model(transitions,
            #                                                    affordance_predictions,
            #                                                    agent_pos_tp1)
            # ptrain = num_elem / float(len(agent_pos_tp1))  # Percentage masked
        else:
            model_loss = [-1.]

        # logs["aff_loss"] = aff_loss
        logs["aff_loss"] = [aff_loss[0]]
        logs["aff_loss_aff"] = [aff_loss[1]]
        logs["aff_loss_NOaff"] = [aff_loss[2]]
        logs["model_loss"] = model_loss
        logs["model_batch_affordable"] = [ptrain]
        logs["model_acc"] = [acc]

        return logs

    def _collect_experiences_step(self, frame_id: int, obs: List[dict], reward: List[float],
                                  done: List[bool], info: List[dict],
                                  prev_obs: DictList, model_result: dict):
        """
            Process new env step return [obs, reward, done, info].
            action was taken by processing previous obs - preprocessed_obs > model_result
        """
        ic_reward_each_step = self.ic_reward_each_step
        envs_buffer = self._envs_buffer
        reset_memory_on_new_op = self._reset_memory_on_new_op
        reset_memory_on_ic = self._reset_memory_on_ic

        # -- Setup option ID
        prev_op = [0] * self.num_procs if model_result is None else \
            model_result["crt_op_idx"].cpu().numpy()

        for ix in range(len(obs)):
            obs[ix]["prev_option"] = prev_op[ix]

        if frame_id >= 0:
            self.model_results[frame_id] = model_result

        if model_result is not None:
            # Reset mem and mask for new op
            true_used_mask = self.mask.clone()

            if reset_memory_on_new_op:
                # Set mask after using it to include new option selections
                # -- we delay - in order to not select new option again
                if self._next_mask is not None:
                    self.mask = self._next_mask

                # calculate new mask for future use
                next_done = torch.tensor(done, device=self.device, dtype=torch.bool)
                self._next_mask = 1. - (model_result["new_op_mask"] | next_done).float()

                # Reset next step memory directly (for the moment not the mask)
                new_op_mask = 1. - model_result["new_op_mask"].float()
                model_result["memory"] = model_result["memory"] * new_op_mask.unsqueeze(1)

            # Create experience buffer
            res_ms = DictList({k: v for k, v in model_result.items() if k != "dist"})

            model_result["termination_target"] = t_t = torch.zeros(len(reward), device=self.device)

            for ix in range(len(done)):
                step_datas = reward[ix], done[ix], info[ix], obs[ix]

                # No env reward
                reward[ix] = 0

                step_data = Experience(prev_obs[ix], *step_datas, res_ms[ix])
                envs_buffer[ix].append(step_data)
                op_id = envs_buffer[ix][0].model_result.crt_op_idx

                # Step 21: Calculate intent completion
                # ----------------------

                new_trajectory = None
                new_tr_exp = None
                true_done = step_data.done
                if step_data.done:  # End trajectory because of env reset
                    fake_next_obs = Experience(
                        self.preprocess_obss([envs_buffer[ix][-1].info["last_obs"]],
                                             device=self.device),
                        None, None, None, None, None
                    )
                    test_tr = envs_buffer[ix] + [fake_next_obs]

                    new_trajectory = self._add_new_trajectory(envs_buffer[ix])
                    ic = self.ic_m.intents_completion(test_tr)
                else:
                    fake_next_obs = Experience(
                        self.preprocess_obss([obs[ix]], device=self.device),
                        None, None, None, None, None
                    )
                    test_tr = envs_buffer[ix] + [fake_next_obs]

                    ic = self.ic_m.intents_completion(test_tr)

                    # End trajectory because of intent achieved / max_horizon reached
                    # Change to resetting option if any option reaches intent
                    if np.any(ic >= self.intent_achieved_th) or \
                            len(envs_buffer[ix]) >= self.intent_max_horizon:
                        new_trajectory = self._add_new_trajectory(envs_buffer[ix])

                        # Reset options
                        if reset_memory_on_ic:
                            step_data.done = True
                            done[ix] = True

                    # End trajectory if new option selected (-> termination signal)
                    if reset_memory_on_new_op and res_ms[ix].new_op_mask and \
                            true_used_mask[ix] == 1. and new_trajectory is None:
                        new_tr_exp = envs_buffer[ix][-1]
                        envs_buffer[ix] = envs_buffer[ix][:-1]
                        new_trajectory = self._add_new_trajectory(envs_buffer[ix])

                if new_trajectory is not None:
                    # Register new trajectory
                    self._crt_finished_trajectories.append(new_trajectory)

                    if new_tr_exp is not None:
                        envs_buffer[ix] = [new_tr_exp]
                    else:
                        envs_buffer[ix] = []

                    self._crt_finished_trajectories_info.append(dict({
                        "intent_completion": ic,
                        "option": op_id.item()
                    }))

                if ic_reward_each_step:
                    # Add intrinsic reward based on intent completion of the option used
                    # Entire trajectory should have been run with the same option id
                    if self.ic_reward_negative:
                        reward[ix] += ic[op_id] - (ic.sum() - ic[op_id])
                    else:
                        reward[ix] += ic[op_id]

                # Save all partial intent completed
                step_data.ic = ic

                # Step 2.2: Calculate termination signal
                # ----------------------
                # TODO HARDCODED Should check if intent is calculated based on last observation
                t_t[ix] = 0 if true_done else ic[op_id]

    def _train_step_options(self, exps: DictList):
        """ Training PPO """
        self._update_step += 1

        # -- Training config variables
        model = self.model
        optimizer = self.optimizer_op
        recurrence = self.recurrence
        recurrent = self.recurrent
        entropy_coef = self.entropy_coef
        value_loss_coef = self.value_loss_coef
        termination_loss_coef = self.termination_loss_coef
        max_grad_norm = self.max_grad_norm
        clip_eps = self.clip_eps
        batch_size = self.batch_size
        memory = None  # type: torch.Tensor

        t_criterion = torch.nn.BCELoss(reduction="none")

        # -- Initialize log values
        log_entropies = []
        log_values = []
        log_policy_losses = []
        log_value_losses = []
        log_termination_losses = []
        log_termination_n_losses = []
        log_termination_p_losses = []
        log_grad_norms = []
        logs = dict()

        for epoch_no in range(self.epochs):
            for inds in self._get_batches_starting_indexes(batch_size, shift_start_idx=False):

                # -- Initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_termination_loss = 0
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

                    if termination_loss_coef > 0:
                        terminations = res_m["op_terminate"]
                        tgt = sb.termination_target.unsqueeze(1)
                        t_loss = t_criterion(terminations, tgt)

                        # Partial loss
                        _p = sb.termination_target > 0
                        t_p_loss = t_loss[_p]
                        t_n_loss = t_loss[~_p]
                        t_loss = t_loss.mean()

                        loss += t_loss * termination_loss_coef
                    else:
                        t_loss, t_p_loss, t_n_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1)

                    # -- Update batch values
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_termination_loss += t_loss.item()
                    log_termination_p_losses.append(t_p_loss.detach())
                    log_termination_n_losses.append(t_n_loss.detach())
                    batch_loss += loss

                    # -- Update memories for next epoch
                    if recurrent and i < recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # -- Update batch values
                batch_entropy /= recurrence
                batch_value /= recurrence
                batch_policy_loss /= recurrence
                batch_value_loss /= recurrence
                batch_termination_loss /= recurrence
                batch_loss /= recurrence

                # -- Update actor-critic
                optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if
                                p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                # -- Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_termination_losses.append(batch_termination_loss)
                log_grad_norms.append(grad_norm)

        # -- Log some values
        logs["entropy"] = [np.mean(log_entropies)]
        logs["value"] = [np.mean(log_values)]
        logs["policy_loss"] = [np.mean(log_policy_losses)]
        logs["value_loss"] = [np.mean(log_value_losses)]
        logs["termination_loss"] = [np.mean(log_termination_losses)]
        logs["grad_norm"] = [np.mean(log_grad_norms)]

        logs["termination_p_loss"] = [torch.cat(log_termination_p_losses).mean().item()]
        logs["termination_n_loss"] = [torch.cat(log_termination_n_losses).mean().item()]

        return logs

    def _process_actions_for_step(self, actions, model_results):
        if self.send_intent_to_env:  # TODO Hack works for num intents < 10
            crt_op = model_results["crt_op_idx"] + 1
            new_op = model_results["new_op_mask"]
            new_a = (crt_op * 100 + new_op * 10 + actions).cpu().numpy()
            return new_a
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

    def _collect_experiences_finished(self) -> dict:
        return dict()

    def get_checkpoint(self) -> dict:
        chkp = dict({
            "optimizer_aff": self.optimizer_aff.state_dict(),
            "optimizer_op": self.optimizer_op.state_dict(),
        })
        if self.use_model:
            chkp["optimizer_tr_model"] = self.optimizer_tr_model.state_dict()

        return chkp

    def load_checkpoint(self, agent_data: dict) -> None:
        if "optimizer_aff" in agent_data:
            self.optimizer_aff.load_state_dict(agent_data["optimizer_aff"])
        if "optimizer_tr_model" in agent_data:
            self.optimizer_tr_model.load_state_dict(agent_data["optimizer_tr_model"])
        if "optimizer_op" in agent_data:
            self.optimizer_op.load_state_dict(agent_data["optimizer_op"])

    def evaluate(self, eval_key=None) -> dict:
        # -- Evaluation config variables
        env = self.eval_envs
        eval_episodes = self.eval_episodes
        preprocess_obss = self.preprocess_obss
        device = self.device
        recurrent = self.recurrent
        model = self.model
        rewards = self.eval_rewards
        mask = self.eval_mask.fill_(1)
        protect_env_data = self._protect_env_data
        num_options = self.model.num_options

        memory = None
        obs = env.reset()

        num_procs = len(obs)
        # -- Setup option ID
        prev_op = [0] * num_procs

        for ix in range(len(obs)):
            obs[ix]["prev_option"] = prev_op[ix]

        if recurrent:
            memory = self.eval_memory
            memory.zero_()

        # -- Initialize log values
        rewards.zero_()

        aff_op_cnt = 0
        successfull_op = 0
        bad_collected = 0
        op_len = []

        mask.fill_(0)

        envs_tr = [[]] * num_procs

        # Wait for eval_episodes to finish
        while aff_op_cnt < eval_episodes:
            # -- Run eva environment steps

            for ix in range(len(obs)):
                envs_tr[ix].append(copy.deepcopy(obs[ix]))

            preprocessed_obs = preprocess_obss(obs, device=device)

            # get new ids
            new_option_mask = mask == 0
            num_new_op = new_option_mask.sum()
            if num_new_op > 0:
                op_idx = torch.multinomial(preprocessed_obs.available_obj.float(), 1)
                preprocessed_obs.prev_option[new_option_mask] = op_idx.squeeze(1)[new_option_mask]

            with torch.no_grad():
                if recurrent:
                    res_m = model(preprocessed_obs, force_no_interest=True, mask=None,
                                  memory=memory * mask)
                else:
                    res_m = model(preprocessed_obs, force_no_interest=True, mask=None)

                action, act_log_prob, value = res_m["actions"], \
                                              res_m["act_log_probs"], \
                                              res_m["values"]

            next_obs, reward, done, info = env.step(self._process_actions_for_step(action, res_m))

            if not protect_env_data:
                next_obs, reward, done, info = list(next_obs), list(reward), list(done), list(info)

            mask = (torch.tensor(1.) - torch.tensor(done, device=device, dtype=torch.float))

            # reset option and count collected
            prev_op = res_m["crt_op_idx"].cpu().numpy()
            for ix in range(len(next_obs)):
                next_obs[ix]["prev_option"] = prev_op[ix]
                if done[ix]:
                    collected = info[ix]["last_obs"]["collected"]
                else:
                    collected = next_obs[ix]["collected"]

                if done[ix] or len(envs_tr[ix]) > self.intent_max_horizon or collected != -1:

                    # Reset OP
                    mask[ix] = 0.
                    crt_tr = envs_tr[ix]
                    envs_tr[ix] = []
                    op_idx = prev_op[ix]
                    op_len.append(len(crt_tr))

                    if crt_tr[0]["available_obj"][op_idx]:
                        aff_op_cnt += 1
                    if collected == prev_op[ix]:
                        successfull_op += 1

                    elif collected != -1:
                        bad_collected += 1

            obs = next_obs
        return {"eval_r": [successfull_op / float(aff_op_cnt)]}

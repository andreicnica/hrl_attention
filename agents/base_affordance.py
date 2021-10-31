from typing import List
import logging
import torch

logger = logging.getLogger(__name__)

from utils.dictlist import DictList
from utils.logging_utils import LogCfg
from agents.base_trajectory_buffer import TrajectoryBufferBase
from utils import get_intent_completion_method
from utils.data_primitives import Experience


class BaseAffordances(TrajectoryBufferBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cfg

        # -- Load configs specific to this agent
        self.intent_size = cfg.intent_size
        self.batch_trajectories = cfg.batch_trajectories
        self.use_affordance_to_mask_model = cfg.use_affordance_to_mask_model
        self.affordance_mask_threshold = cfg.affordance_mask_threshold
        self.use_affordances = cfg.use_affordances
        self.use_model = cfg.use_model

        self.ic_m = get_intent_completion_method(cfg.intent_completion_method, cfg)

        # -- Optimizer transition model
        optimizer = getattr(cfg, "optimizer_tr_model", "SGD")
        optimizer_args = getattr(cfg, "optimizer_args_tr_model", {})
        optimizer_args = vars(optimizer_args)
        self.optimizer_tr_model = getattr(torch.optim, optimizer)(
            self.model.transition_model.parameters(), **optimizer_args
        )

        # -- Optimizer for aff model
        optimizer = getattr(cfg, "optimizer_aff", "SGD")
        optimizer_args = getattr(cfg, "optimizer_args_aff", {})
        optimizer_args = vars(optimizer_args)
        self.optimizer_aff = getattr(torch.optim, optimizer)(
            self.model.affordance_network.parameters(), **optimizer_args
        )

        # Add log for update_step: to print & write to csv
        self._logs_train_config += [
            LogCfg("trajectories_cnt", False, "l", "t", "", True, False),
            LogCfg("aff_loss", False, "l", "aff_loss", ":.4f", True, True),
            LogCfg("aff_loss_aff", False, "l", "aff", ":.4f", True, True),
            LogCfg("aff_loss_NOaff", False, "l", "NOaff", ":.4f", True, True),
            LogCfg("model_loss", False, "l", "model_loss", ":.4f", True, True),
            LogCfg("model_batch_affordable", False, "l", "mAFF%", ":.4f", True, True),
            LogCfg("model_acc", False, "l", "model_acc", ":.4f", True, True),
        ]

        # # Activate evaluator
        # self.has_evaluator = False

    def _train_step_affordances(self, trajectory):
        """ Train affordance network.

        Args:
            trajectory: a 4-tuple containing the batch of state, action, state' and
             intent target.
             affordance_network: The affordance network.
             affordance_optimizer: The optimizer to use for training.
        """
        affordance_network = self.model.affordance_network
        affordance_optimizer = self.optimizer_aff

        s_t, a_t, s_tp1, _, intent_target = trajectory

        preds = affordance_network(s_t, a_t)
        unshaped_preds = preds

        no_aff = intent_target.sum(dim=1) == 0
        loss_aff = torch.nn.functional.binary_cross_entropy(preds[~no_aff], intent_target[~no_aff])
        loss_no_aff = torch.nn.functional.binary_cross_entropy(preds[no_aff], intent_target[no_aff])

        loss = torch.nn.functional.binary_cross_entropy(preds, intent_target)
        total_loss = loss
        affordance_optimizer.zero_grad()
        total_loss.backward()
        affordance_optimizer.step()

        return [total_loss.item(), loss_aff.item(), loss_no_aff.item()], unshaped_preds

    def _train_step_model(self, trajectory, affordances, agent_pos_tp1):
        """Train model network."""
        use_affordance_to_mask_model = self.use_affordance_to_mask_model
        affordance_mask_threshold = self.affordance_mask_threshold
        model_network = self.model.transition_model
        model_optimizer = self.optimizer_tr_model

        s_t, a_t, s_tp1, _, _ = trajectory
        transition_model_preds = model_network(s_t, a_t)

        # Transform the agent-pos
        width = self.model.image_size[0]
        agent_pos_target = (agent_pos_tp1[:, 0]*width + agent_pos_tp1[:, 1]).long()

        reconstruction_loss = torch.nn.CrossEntropyLoss(reduction="none")
        reconstruction_loss = reconstruction_loss(transition_model_preds, agent_pos_target)

        correct = transition_model_preds.max(dim=1).indices == agent_pos_target

        if use_affordance_to_mask_model:
            # Check if at least one intent is affordable.
            masks_per_intent = torch.ge(affordances, affordance_mask_threshold)
            masks_per_transition = torch.gt(torch.sum(masks_per_intent, dim=1), 0)

            reconstruction_loss = reconstruction_loss * masks_per_transition.float()
            correct = correct[masks_per_transition]

        acc = correct.sum() / float(len(correct))

        total_loss = reconstruction_loss.mean()

        model_optimizer.zero_grad()
        total_loss.backward()
        model_optimizer.step()
        return [total_loss.item()], acc, len(correct)

    def update_parameters(self) -> dict:
        """  Implement agent training  """
        model = self.model
        use_affordances = self.use_affordances
        use_model = self.use_model
        num_procs = self.num_procs

        batch_trajectories = self.batch_trajectories

        logs = dict()

        # ===============================================
        # Train loop
        # ===============================================

        # Step 1: Collect data.
        # ----------------------
        step = 0
        new_tr = []  # type: List[List[Experience]]

        while len(new_tr) < batch_trajectories:
            tr, log_collect = self._collect_experiences()  # 1 step in n x environments
            new_tr += tr
            step += 1

        # Calculate intent completion for primitives
        s_t, a_t, s_tp1, completion, agent_pos_tp1 = [], [], [], [], []

        for tr in new_tr:
            for ix in range(len(tr) - 1):
                action = tr[ix].model_result.actions.item()
                completion.append(self.ic_m.intents_completion(tr[ix: ix+2]))

                agent_pos_tp1.append(tr[ix].next_obs.agent_pos)

                obs = torch.transpose(torch.transpose(tr[ix].obs.image, 0, 2), 1, 2).contiguous()
                obst1 = torch.transpose(torch.transpose(tr[ix].next_obs.image, 0, 2), 1, 2).contiguous()
                s_t.append(obs)
                s_tp1.append(obst1)
                a_t.append(action)

        # Process trajectories and change format
        s_t = torch.stack(s_t, dim=0)
        s_tp1 = torch.stack(s_tp1, dim=0)
        a_t = torch.tensor(a_t, device=self.device).float()
        completion = torch.tensor(completion, device=self.device).float()
        agent_pos_tp1 = torch.stack(agent_pos_tp1, dim=0)
        transitions = (s_t, a_t, s_tp1, None, completion)

        logs["trajectories_cnt"] = [len(new_tr)]
        logs["num_frames"] = [step*num_procs]

        # Step 2: Train Affordance model
        # ----------------------
        if use_affordances:
            aff_loss, affordance_predictions = self._train_step_affordances(transitions)
        else:
            affordance_predictions = torch.FloatTensor([0.0])  # Basically a none.
            aff_loss = [-1.]

        # Step 3: Train Probabilistic transition model
        # ----------------------

        # Train transition model.
        acc, ptrain = 0., 0.
        if use_model:
            model_loss, acc, num_elem = self._train_step_model(transitions,
                                                               affordance_predictions,
                                                               agent_pos_tp1)
            ptrain = num_elem / float(len(agent_pos_tp1))  # Percentage masked
        else:
            model_loss = [-1.]

        logs["aff_loss"] = [aff_loss[0]]
        logs["aff_loss_aff"] = [aff_loss[1]]
        logs["aff_loss_NOaff"] = [aff_loss[2]]
        logs["model_loss"] = model_loss
        logs["model_batch_affordable"] = [ptrain]
        logs["model_acc"] = [acc]

        return logs

    def get_checkpoint(self) -> dict:
        return dict({"optimizer_aff": self.optimizer_aff.state_dict(),
                     "optimizer_tr_model": self.optimizer_tr_model.state_dict()})

    def load_checkpoint(self, agent_data: dict) -> None:
        if "optimizer_aff" in agent_data:
            self.optimizer_aff.load_state_dict(agent_data["optimizer_aff"])
        if "optimizer_tr_model" in agent_data:
            self.optimizer_tr_model.load_state_dict(agent_data["optimizer_tr_model"])

    def _collect_experiences_step(self, frame_id: int, obs: List[DictList], reward: List[float],
                                  done: List[bool], info: List[dict], prev_obs: DictList,
                                  model_result: dict):
        pass

    def _collect_experiences_finished(self) -> dict:
        return dict()

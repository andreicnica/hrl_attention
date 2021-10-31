from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from typing import Optional, Tuple
import gym
import numpy as np
import logging

logger = logging.getLogger(__name__)

from models.utils import initialize_parameters
from utils.dictlist import DictList
from .base import ModelBase
from models.extractor_cnn_v2 import CNNFeatureExtractorV2


class OcSimple(nn.Module, ModelBase):
    def __init__(self, cfg: Namespace, obs_space: dict, action_space: gym.spaces):
        super().__init__()

        # -- Load config
        self.num_options = num_options = getattr(cfg, "num_options", 4)
        self.eps_greedy = getattr(cfg, "eps_greedy", 0.05)
        self.terminate_th = getattr(cfg, "terminate_th", 0.5)
        self.recurrent = cfg.recurrent
        device = cfg.device

        hidden_size = cfg.hidden_size[-1]
        num_actions = action_space.n

        self.feature_extractor = CNNFeatureExtractorV2(cfg, obs_space, action_space)
        self.feature_size = crt_size = self.feature_extractor.embedding_size

        logger.info(f"OBS space {obs_space}")
        # Option values
        self.option_qvalues = nn.Sequential(
            nn.Linear(crt_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_options)
        )

        # Option termination function
        self.termination = nn.ModuleList([nn.Sequential(
            nn.Linear(crt_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        ) for _ in range(num_options)])

        # Option policies
        self.option_policy = nn.ModuleList([nn.Sequential(
            nn.Linear(crt_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        ) for _ in range(num_options)])

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return self.feature_extractor.memory_size

    def options_logits(self, embedding: torch.Tensor, option_idx: torch.Tensor):
        batch = option_idx.size(0)
        all_op_logits = [op(embedding) for op in self.option_policy]
        all_op_logits = torch.stack(all_op_logits, dim=1)
        op_logits = all_op_logits.gather(
            1, option_idx.view(batch, 1, 1).expand(batch, 1, all_op_logits.size(-1))
        ).squeeze(1)

        return op_logits, all_op_logits

    def forward(self, obs: DictList, memory: torch.Tensor = None, mask: torch.Tensor = None,
                eps_greedy: float = None, **kwargs):

        eps_g = self.eps_greedy if eps_greedy is None else eps_greedy
        device = obs.image.device
        num_options = self.num_options
        terminate_th = self.terminate_th

        crt_op_idx = obs.prev_option

        # -- Reshape image to be CxWxH and extract common features
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3).contiguous()
        embedding, memory = self.feature_extractor(x, memory=memory)

        # -- Choose options (expected done_mask to equal 0 when new episode starts
        op_q = self.option_qvalues(embedding)
        op_terminate, op_terminations = self.option_terminations(embedding, crt_op_idx)

        new_option_mask = None
        # Do not change option if no mask is available e.g. (training run)
        if mask is not None:
            # Get eps-greedy option
            new_option_mask = ((op_terminate > terminate_th).squeeze(1) | (mask == 0))
            if new_option_mask.sum() > 0:
                op_idx = op_q.argmax(dim=1)  # Choose option greedy

                rand_mask = torch.rand((new_option_mask.sum(),), device=device) < eps_g
                rand_op = torch.randint(num_options, (rand_mask.sum(),), device=device)

                # Change option index only for selected ones
                op_idx = op_idx[new_option_mask]
                op_idx[rand_mask] = rand_op
                crt_op_idx[new_option_mask] = op_idx

        # -- Get actions for options
        selected_op_logits, _ = self.options_logits(embedding, crt_op_idx)
        dist = Categorical(logits=F.log_softmax(selected_op_logits, dim=1))

        action_sample = dist.sample()
        ret_m = dict({
            "actions": action_sample,
            "act_log_probs": dist.log_prob(action_sample),
            "dist": dist,
            "values": op_q.gather(1, crt_op_idx.unsqueeze(1)).squeeze(1),
            "crt_op_idx": crt_op_idx,
            "op_terminations": op_terminations,
            "op_terminate": op_terminate,
            "op_qvalues": op_q,
            "prev_op": obs.prev_option,
            "new_op_mask": new_option_mask,

        })

        if self.recurrent:
            ret_m["memory"] = memory

        return ret_m

    def option_terminations(self, embedding: torch.Tensor, option_idx: torch.Tensor):
        batch = option_idx.size(0)
        all_terminations = [ttt(embedding) for ttt in self.termination]
        all_terminations = torch.stack(all_terminations, dim=1)
        terminations = all_terminations.gather(
            1, option_idx.view(batch, 1, 1).expand(batch, 1, all_terminations.size(-1))
        ).squeeze(1)

        return terminations, all_terminations

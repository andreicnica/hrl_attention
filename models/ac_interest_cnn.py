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


class AcInterestCNN(nn.Module, ModelBase):
    def __init__(self, cfg: Namespace, obs_space: dict, action_space: gym.spaces):
        super().__init__()

        # -- Load config
        self.recurrent = recurrent = cfg.recurrent
        self.use_text = cfg.use_text
        self.use_interest = getattr(cfg, "use_interest", False)

        hidden_size = cfg.hidden_size

        assert len(hidden_size) > 0

        kernels = getattr(cfg, "kernels", [16, 32, 64])
        k_sizes = getattr(cfg, "kernel_sizes", [5, 5, 3])
        s_sizes = getattr(cfg, "stride_sizes", [3, 3, 1])

        if recurrent:
            self.memory_type = memory_type = cfg.memory_type
            self._memory_size = memory_size = cfg.memory_size
        else:
            self._memory_size = None

        logger.info(f"OBS space {obs_space}")
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        ch = obs_space["image"][2]

        # experiment used model
        self.image_conv = nn.Sequential(
            nn.Conv2d(ch, kernels[0], k_sizes[0], s_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[0], kernels[1], k_sizes[1], s_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[1], kernels[2], k_sizes[2], s_sizes[2]),
            nn.ReLU(inplace=True)
        )

        out_conv_size = self.image_conv(torch.rand((1, obs_space["image"][2], n, m))).size()
        out_feat_size = int(np.prod(out_conv_size))

        self.image_embedding_size = out_feat_size

        # Reshape for linear input
        self.fc1 = nn.Sequential(
            nn.Linear(out_feat_size, hidden_size[0]),
            nn.ReLU(inplace=True),
        )

        hidden_size = hidden_size if len(hidden_size) == 1 else hidden_size[1:]
        crt_size = hidden_size[0]
        hidden_size = hidden_size[0] if len(hidden_size) == 1 else hidden_size[1]

        # Define memory
        if recurrent:
            if memory_type == "LSTM":
                self.memory_rnn = nn.LSTMCell(crt_size, memory_size)
            else:
                self.memory_rnn = nn.GRUCell(crt_size, memory_size)

            crt_size = memory_size

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size,
                                   batch_first=True)

        # Resize image embedding
        self.embedding_size = crt_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        self.fc2_val = nn.Sequential(
            nn.Linear(self.embedding_size, hidden_size),
            nn.ReLU(),
        )

        self.fc2_act = nn.Sequential(
            nn.Linear(self.embedding_size, hidden_size),
            nn.ReLU(),
        )

        # Define heads
        self.vf = nn.Linear(hidden_size, 1)
        self.pd = nn.Linear(hidden_size, action_space.n)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        if self.recurrent:
            if self.memory_type == "LSTM":
                return 2 * self._memory_size
            else:
                return self._memory_size

        return self._memory_size

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def forward(self, obs: DictList, memory: torch.Tensor = None, no_interest=False, **kwargs):
        # Reshape image to be CxWxH
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3).contiguous()
        x = self.image_conv(x)
        x = x.flatten(1)
        x = self.fc1(x)

        if self.recurrent:
            if self.memory_type == "LSTM":
                hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
                hidden = self.memory_rnn(x, hidden)  # type: Tuple[torch.Tensor]
                embedding = hidden[0]
                memory = torch.cat(hidden, dim=1)
            else:
                hidden = memory  # type: Optional[torch.Tensor]
                hidden = self.memory_rnn(x, hidden)
                embedding = hidden
                memory = hidden
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        val = self.fc2_val(embedding)
        act = self.fc2_act(embedding)

        vpred = self.vf(val).squeeze(1)

        pd = self.pd(act)

        pd = F.softmax(pd, dim=1)

        if self.use_interest and not no_interest:
            interest = obs.interest
            pd = pd * interest + 0.000003

            pdsum = pd.sum(dim=1)
            pd /= pdsum.unsqueeze(1)
            pd.clamp_(0.00000003, 0.99999997)  # numerical stability

        dist = Categorical(logits=torch.log(pd))

        action_sample = dist.sample()
        ret_m = dict({"actions": action_sample, "act_log_probs": dist.log_prob(action_sample),
                      "dist": dist, "values": vpred})

        if self.recurrent:
            ret_m["memory"] = memory

        return ret_m


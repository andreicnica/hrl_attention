import torch
import torch.nn as nn
from argparse import Namespace
import numpy as np
import gym


class CNNFeatureExtractor(nn.Module):
    def __init__(self, cfg: Namespace, obs_space: dict, action_space: gym.spaces):
        super().__init__()

        # -- Load config
        self.recurrent = recurrent = cfg.recurrent

        hidden_size = cfg.hidden_size
        kernels = getattr(cfg, "kernels", [16, 32, 64])
        k_sizes = getattr(cfg, "kernel_sizes", [5, 5, 3])
        s_sizes = getattr(cfg, "stride_sizes", [3, 3, 1])
        paddings = getattr(cfg, "padding", [0, 0, 0])

        if recurrent:
            self.memory_type = memory_type = getattr(cfg, "memory_type", "GRU")
            self._memory_size = memory_size = getattr(cfg, "memory_size", 128)
        else:
            self._memory_size = None

        assert len(obs_space["image"]) == 3, "Set channels on dim 2"

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        ch = obs_space["image"][2]

        # experiment used model
        self.image_conv = nn.Sequential(
            nn.Conv2d(ch, kernels[0], k_sizes[0], s_sizes[0], padding=paddings[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[0], kernels[1], k_sizes[1], s_sizes[1], padding=paddings[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[1], kernels[2], k_sizes[2], s_sizes[2], padding=paddings[2]),
            nn.ReLU(inplace=True)
        )

        out_conv_size = self.image_conv(torch.rand((1, obs_space["image"][2], n, m))).size()
        out_feat_size = int(np.prod(out_conv_size))

        # Reshape for linear input
        self.fc1 = nn.Sequential(
            nn.Linear(out_feat_size, hidden_size[0]),
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

        self._embedding_size = crt_size

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def memory_size(self):
        if self.recurrent:
            if self.memory_type == "LSTM":
                return 2 * self._memory_size
            else:
                return self._memory_size

        return self._memory_size

    def forward(self, obs, memory: torch.Tensor = None):
        # Reshape image to be CxWxH
        x = self.image_conv(obs)
        x = x.flatten(1)
        x = self.fc1(x)

        memory = None

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

        return embedding, memory

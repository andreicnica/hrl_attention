from argparse import Namespace
import torch
import torch.nn as nn
import gym
import logging
import numpy as np
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from copy import deepcopy

logger = logging.getLogger(__name__)

from utils.dictlist import DictList
from .base import ModelBase
from models.oc_simple import OcSimple
from models.extractor_cnn_v2 import CNNFeatureExtractorV2

EPS = 0.000003

def concat_state_with_action(states: torch.Tensor, actions: torch.Tensor, num_actions: int):
    b, ch, w, h = states.size()
    act_emb = torch.zeros(b, num_actions, device=states.device)
    act_emb.scatter_(1, actions.unsqueeze(1).long(), 1)
    act_emb = act_emb.view(b, num_actions, 1, 1).expand(b, num_actions, w, h)

    x = torch.cat([states, act_emb], dim=1)
    return x


class AffordanceCNN(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super().__init__()
        self.num_options = num_options = getattr(cfg, "num_options", 4)
        intent_size = 1  # cfg.intent_size # TODO test
        use_sigmoid = getattr(cfg, "use_sigmoid", True)
        affordance_recurrent = getattr(cfg, "affordance_recurrent", False)

        new_obs_space = deepcopy(obs_space)
        new_obs_space["image"] = list(new_obs_space["image"])
        new_obs_space["image"][2] += num_options

        aff_cfg = deepcopy(cfg)
        aff_cfg.recurrent = affordance_recurrent
        self.feature_extractor = CNNFeatureExtractorV2(aff_cfg, new_obs_space, action_space)
        self.feature_size = crt_size = self.feature_extractor.embedding_size

        if use_sigmoid:
            self.affordance_network = nn.Sequential(
                nn.Linear(crt_size, intent_size),
                nn.Sigmoid()
            )
        else:
            self.affordance_network = nn.Sequential(
                nn.Linear(crt_size, intent_size)
            )

    def forward(self, states, option_ids, **kwargs):
        x = concat_state_with_action(states, option_ids, self.num_options)

        x, memory = self.feature_extractor(x)
        x = x.flatten(1)
        x = self.affordance_network(x)
        return x


class OptionsMultistep(OcSimple):
    def __init__(self, cfg: Namespace, obs_space: dict, action_space: gym.spaces):
        super().__init__(cfg, obs_space, action_space)
        self.sample_attention_op = getattr(cfg, "sample_attention_op", False)
        self._env_max_room = getattr(cfg, "env_max_room", 1)  # TODO HACK for ground truth room
        self._room_gt_interest = getattr(cfg, "room_gt_interest", False)  # TODO HACK for ground truth room

    def forward(self, obs: DictList, memory: torch.Tensor = None, mask: torch.Tensor = None,
                interest: torch.Tensor = None, **kwargs):

        device = obs.image.device
        num_options = self.num_options
        terminate_th = self.terminate_th
        # Correct for old checkpoints
        attention = self.sample_attention_op if hasattr(self, "sample_attention_op") else \
            self.sample_affordable_op
        crt_op_idx = obs.prev_option

        # -- Reshape image to be CxWxH and extract common features
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3).contiguous()
        embedding, memory = self.feature_extractor(x, memory=memory)

        op_q = self.option_qvalues(embedding)
        op_terminate, op_terminations = self.option_terminations(embedding, crt_op_idx)

        random_sample = not attention

        new_option_mask = None
        # Do not change option if no mask is available e.g. (training run)
        if mask is not None:
            # -- Choose options (expected done_mask to equal 0 when new episode starts
            new_option_mask = ((op_terminate > terminate_th).squeeze(1) | (mask == 0))
            num_new_op = new_option_mask.sum()
            if num_new_op > 0:
                if random_sample or interest is None:
                    op_idx = torch.randint(num_options, (num_new_op,), device=device)
                else:
                    if not self._room_gt_interest:
                        op_idx = torch.multinomial(interest[new_option_mask], 1).squeeze(1)
                    else:
                        # TODO Hack for ground truth room
                        # Choose from achievable options - HACK for change room case
                        max_room = self._env_max_room
                        room = obs.room_pos[new_option_mask]
                        aff = torch.ones(num_new_op, num_options, device=device)
                        aff[:, 0].copy_(room[:, 0] > 0)
                        aff[:, 1].copy_(room[:, 1] > 0)
                        aff[:, 2].copy_(room[:, 0] < max_room)
                        aff[:, 3].copy_(room[:, 1] < max_room)
                        op_idx = torch.multinomial(aff / aff.sum(dim=1).unsqueeze(1), 1).squeeze(1)

                # Change option index only for selected ones
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


class AffMultiStepNet(nn.Module, ModelBase):
    def __init__(self, cfg: Namespace, obs_space: dict, action_space: gym.spaces):
        super().__init__()

        self.num_actions = action_space.n
        self.use_affordances = cfg.use_affordances
        self.sample_attention_op = cfg.sample_attention_op
        self.sample_interest = getattr(cfg, "sample_interest", True)
        self.affordance_th = getattr(cfg, "affordance_th", -1.)
        self.num_options = num_options = getattr(cfg, "num_options", 4)
        self.interest_temp = getattr(cfg, "interest_temp", 1.)
        self.recurrent = cfg.recurrent

        n, m, ch = obs_space["image"]
        self.image_size = (n, m, ch)

        # Init models
        self.options = OptionsMultistep(cfg, obs_space, action_space)
        self.affordance_network = AffordanceCNN(cfg, obs_space, action_space)
        self.transition_model = None  # TransitionModelCNN(cfg, (n, m, ch), action_space.n)

        logger.info(f"Observation shapes: {obs_space}")

    def forward(self, obs: DictList, force_no_interest=False, **kwargs):
        # Run only options
        interest = None
        ret_aff = None
        num_options = self.num_options
        interest_a = None

        if not force_no_interest and (self.use_affordances or self.sample_attention_op):
            ret_aff = self.get_interest(obs, **kwargs)
            interest = ret_aff["interest"]
            op_aff = ret_aff["op_aff"]
            if getattr(self, "sample_interest", True):
                interest_a = interest
            else:
                interest_a = (ret_aff["op_aff"] > self.affordance_th).float() + EPS
                interest_sum = interest_a.sum(dim=1)
                interest_a /= interest_sum.unsqueeze(1)

        ret_m = self.options(obs, interest=interest_a, **kwargs)

        if ret_aff is not None:
            ret_m["affordance"] = op_aff
            ret_m["interest"] = interest

        return ret_m

    def get_interest(self, obs, **kwargs):
        interest_temp = getattr(self, "interest_temp", 1.)
        num_options = self.num_options

        # -- Get affordances for all options X intents
        states = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3).contiguous()

        st_size = states.size()
        b_size = st_size[0]

        states = states.unsqueeze(1).expand(b_size, num_options, *st_size[1:]).contiguous()
        states = states.view(-1, *st_size[1:])
        op_idx = torch.arange(num_options, device=states.device).unsqueeze(0) \
            .expand(b_size, num_options).contiguous().view(-1)

        ret_aff = self.affordance_network(states, op_idx, **kwargs)

        # Select the intent corresponding to the given option
        op_aff = ret_aff.view(b_size, num_options)

        interest = F.softmax(op_aff / interest_temp, dim=1)
        return dict({"interest": interest, "op_aff": op_aff})

    @property
    def memory_size(self):
        return self.options.memory_size


if __name__ == "__main__":
    from argparse import Namespace

    batch_size = 256
    num_options = 4
    max_room = 2

    norm_aff_target = 3.
    criterion = F.mse_loss

    cfg_net = Namespace()
    cfg_net.hidden_size = [256]
    cfg_net.num_options = num_options
    cfg_net.intent_size = num_options
    cfg_net.use_sigmoid = False
    cfg_net.affordance_recurrent = False
    cfg_net.kernels = [32, 32, 64]
    cfg_net.kernel_sizes = [5, 5, 5]
    cfg_net.stride_sizes = [1, 1, 1]
    cfg_net.padding = [0, 0, 0]

    obs_space = dict({"image": [19, 19, 3]})

    net = AffordanceCNN(cfg_net, obs_space, None)
    net = net.cuda()

    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.0003)

    # -- Dataset generation
    data_eg = torch.load("otest")
    state = data_eg[0]
    state = torch.transpose(torch.transpose(state, 0, 2), 1, 2).contiguous()

    agent_id = 10. / 10.
    goal_id = 8 / 10.
    room_size = 6

    # make empty state
    state[torch.where(state == agent_id)] = 0.
    state[torch.where(state == goal_id)] = 0.
    empty_state = state.clone().unsqueeze(0).expand(batch_size, *state.size())
    device = state.device

    empty_pos = torch.where(state[0] == 0.1)
    num_empty = len(empty_pos[0])

    pio = [torch.arange(batch_size).long().to(device), torch.zeros(batch_size).long().to(device)]

    def get_batch():
        ag_poss = torch.randint(num_empty, (batch_size,))
        g_poss = torch.randint(num_empty, (batch_size,))

        ag_pos = tuple(pio + [x[ag_poss] for x in empty_pos])
        g_pos = tuple(pio + [x[g_poss] for x in empty_pos])

        data = empty_state.clone()
        data[g_pos] = goal_id
        data[ag_pos] = agent_id

        room = torch.stack(ag_pos[2:]).t()
        room /= room_size

        aff = torch.ones(batch_size, num_options, device=device)
        aff[:, 0].copy_(room[:, 0] > 0)
        aff[:, 1].copy_(room[:, 1] > 0)
        aff[:, 2].copy_(room[:, 0] < max_room)
        aff[:, 3].copy_(room[:, 1] < max_room)
        return data, aff

    epochs = 10000
    for i in range(epochs):
        data, target = get_batch()

        optimizer.zero_grad()

        op_ids = torch.randint(num_options, (batch_size, )).to(device).float()
        outputs = net(data, op_ids)

        mask = torch.zeros_like(target)
        mask.scatter_(1, op_ids.unsqueeze(1).long(), 1)
        target *= mask

        if norm_aff_target > 0.:
            target = target * norm_aff_target - norm_aff_target / 2
            target.detach_()

        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        print(f"e: {i} - loss: {loss.item()}")

import torch
from torch import Tensor, nn
from typing import List
import numpy as np
import torch.distributions as dists

from core.model import bounded_call
from core.layer.utils import get_torch_layers


class IWAEObjective:
    def __init__(self, kl_penalty: float, n: int) -> None:
        self._kl_penalty = kl_penalty
        self.n: int = n
        self.criterion = torch.nn.NLLLoss()

    def calculate(self, model: nn.Module, data: Tensor, target: Tensor, pmin: float = None) -> Tensor:

        log_losses = []

        for i in range(self.n):

            if pmin is not None:
                p_x_g_w = bounded_call(model, data, pmin)
            else:
                p_x_g_w = model(data)

            # log_loss_i = torch.sum(p_x_g_w, dim=1)
            # log_loss_i = self.criterion(p_x_g_w, target)
            temperature = 0.00001
            log_loss_i = dists.Categorical(logits=p_x_g_w).log_prob(target)

            log_p_w_total = 0
            log_q_w_g_x_total = 0
            eps = 1e-6
            norm = False

            for l_name, l in get_torch_layers(model):
                sampled_weight = l._sampled_weight
                sampled_bias = l._sampled_bias

                log_p_w_weight = dists.Normal(l._prior_weight_dist.mu, l._prior_weight_dist.sigma + eps).log_prob(sampled_weight)
                log_p_w_bias = dists.Normal(l._prior_bias_dist.mu, l._prior_bias_dist.sigma + eps).log_prob(sampled_bias)

                if norm:
                    log_p_w_total += (log_p_w_weight.sum() / torch.prod(torch.tensor(log_p_w_weight.shape))
                                      + log_p_w_bias.sum() / torch.prod(torch.tensor(log_p_w_bias.shape)))
                else:
                    log_p_w_total += log_p_w_weight.sum() + log_p_w_bias.sum()

                log_q_w_g_x_weight = dists.Normal(l._weight_dist.mu, l._weight_dist.sigma + eps).log_prob(sampled_weight)
                log_q_w_g_x_bias = dists.Normal(l._bias_dist.mu, l._bias_dist.sigma + eps).log_prob(sampled_bias)

                if norm:
                    log_q_w_g_x_total += (log_q_w_g_x_weight.sum() / torch.prod(torch.tensor(log_q_w_g_x_weight.shape))
                                          + log_q_w_g_x_bias.sum() / torch.prod(torch.tensor(log_q_w_g_x_bias.shape)))
                else:
                    log_q_w_g_x_total += log_q_w_g_x_weight.sum() + log_q_w_g_x_bias.sum()

            log_loss_i = log_loss_i + (log_p_w_total.repeat(len(log_loss_i)) - log_q_w_g_x_total.repeat(len(log_loss_i)))*temperature
            log_losses.append(log_loss_i)
        loss = - (torch.logsumexp(torch.stack(log_losses), dim=0) - np.log(self.n)).mean()
        # loss = -log_losses[0].mean()
        return loss

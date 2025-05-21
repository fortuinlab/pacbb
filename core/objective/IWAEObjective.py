import logging, math
from typing import Dict, Optional

import torch, torch.distributions as dists, wandb
from torch import nn, Tensor

from core.model import bounded_call
from core.layer.utils import get_torch_layers


class IWAEObjective:
    def __init__(self, kl_penalty: float, n: int, temperature: float = 1.0) -> None:
        self.k = n
        self.kl_penalty = kl_penalty      # usually 1 / |D|
        self.temperature = temperature

    # -------- helpers to compute log p(w) and log q(w) -------------------
    @staticmethod
    def _log_prior(model: nn.Module, eps: float = 1e-6) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        s = torch.zeros(1, device=device, dtype=dtype)

        for _, l in get_torch_layers(model):
            s += dists.Normal(l._prior_weight_dist.mu,
                              l._prior_weight_dist.sigma + eps
                              ).log_prob(l._sampled_weight).sum()
            s += dists.Normal(l._prior_bias_dist.mu,
                              l._prior_bias_dist.sigma + eps
                              ).log_prob(l._sampled_bias).sum()
        return s

    @staticmethod
    def _log_post(model: nn.Module, eps: float = 1e-6) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        s = torch.zeros(1, device=device, dtype=dtype)

        for _, l in get_torch_layers(model):
            s += dists.Normal(l._weight_dist.mu,
                              l._weight_dist.sigma + eps
                              ).log_prob(l._sampled_weight).sum()
            s += dists.Normal(l._bias_dist.mu,
                              l._bias_dist.sigma + eps
                              ).log_prob(l._sampled_bias).sum()
        return s

    # --------------------------------------------------------------------
    def calculate(
        self,
        model: nn.Module,
        data: Tensor,
        target: Tensor,
        epoch: int,
        batch_idx: int,
        dataset_size: int,
        pmin: Optional[float] = None,
        wandb_params: Optional[Dict] = None,
    ) -> Tensor:

        batch_size = data.size(0)
        scale = dataset_size / batch_size           # N / |B|
        log_ws = []                                 # list[k] of scalars

        for l in range(self.k):
            # sample w and compute log p(x|w)
            logits = bounded_call(model, data, pmin) if pmin is not None else model(data)
            log_px = dists.Categorical(logits=logits).log_prob(target)   # (batch,)
            log_lik = scale * log_px.sum()                               # scalar

            # global KL part
            kl = (self._log_prior(model) - self._log_post(model)) * self.kl_penalty
            log_w = log_lik + self.temperature * kl                      # scalar
            log_ws.append(log_w)

            # -------------------- per-sample logging --------------------
            if wandb_params and wandb_params.get("log_wandb", False):
                tag = wandb_params["name_wandb"]
                wandb.log({
                    f"{tag}/epoch": epoch,
                    f"{tag}/batch": batch_idx,
                    f"{tag}/sample": l,
                    f"{tag}/log_likelihood": log_lik.detach(),
                    f"{tag}/kl": kl.detach(),
                    f"{tag}/log_weight": log_w.detach(),
                })

        # ----------- PB-IWAE loss (one scalar) ---------------------------
        log_ws_tensor = torch.stack(log_ws)           # (k,)
        loss = -(torch.logsumexp(log_ws_tensor, dim=0) - math.log(self.k))

        # ----------- final logging --------------------------------------
        if wandb_params and wandb_params.get("log_wandb", False):
            wandb.log({f"{wandb_params['name_wandb']}/iwae_loss": loss})

        if batch_idx == 0:
            logging.info(
                f"[Epoch {epoch:03d} | Batch {batch_idx:04d}] "
                f"IWAE-loss {loss.item():.4f} "
                f"| mean log_px {(log_px.mean()).item():.4f} "
                f"| KL {kl.item():.2f}"
            )
        return loss

import numpy as np
import torch
import torch.nn as nn


class ProbabilisticLinearLayer(nn.Module):
    """Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_layer : Linear object
        Linear layer object used to initialise the prior

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(
        self,
        in_features,
        out_features,
        rho_prior,
        prior_dist="gaussian",
        device="cuda",
        init_prior="random",
        init_layer=None,
        fix_mu=False,
        fix_rho=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1 / np.sqrt(in_features)

        # INITIALISE PRIOR
        # this means prior is uninformed
        if not init_layer:
            # initalise prior to zeros and rho_prior
            if init_prior == "zeros":
                bias_mu_prior = torch.zeros(out_features)
                weights_mu_prior = torch.zeros(out_features, in_features)
                weights_rho_prior = torch.ones(out_features, in_features) * rho_prior
                bias_rho_prior = torch.ones(out_features) * rho_prior
            # initialise prior to random weights and rho_prior
            elif init_prior == "random":
                weights_mu_prior = trunc_normal_(
                    torch.Tensor(out_features, in_features),
                    0,
                    sigma_weights,
                    -2 * sigma_weights,
                    2 * sigma_weights,
                )
                bias_mu_prior = torch.zeros(out_features)
                weights_rho_prior = torch.ones(out_features, in_features) * rho_prior
                bias_rho_prior = torch.ones(out_features) * rho_prior
            else:
                raise RuntimeError(f"Wrong type of prior initialisation!")
        # informed prior
        else:
            # if init layer is probabilistic
            if hasattr(init_layer.weight, "rho"):
                weights_mu_prior = init_layer.weight.mu
                bias_mu_prior = init_layer.bias.mu
                weights_rho_prior = init_layer.weight.rho
                bias_rho_prior = init_layer.bias.rho
            # if init layer for prior is not probabilistic
            else:
                weights_mu_prior = init_layer.weight
                bias_mu_prior = init_layer.bias
                weights_rho_prior = torch.ones(out_features, in_features) * rho_prior
                bias_rho_prior = torch.ones(out_features) * rho_prior

        # INITIALISE POSTERIOR
        # WE ASSUME THAT ALWAYS POSTERIOR WILL BE INITIALISED TO PRIOR (UNLESS PRIOR IS INITIALISED TO ALL ZEROS)
        if init_prior == "zeros":
            weights_mu_init = trunc_normal_(
                torch.Tensor(out_features, in_features),
                0,
                sigma_weights,
                -2 * sigma_weights,
                2 * sigma_weights,
            )
            bias_mu_init = torch.zeros(out_features)
            weights_rho_init = torch.ones(out_features, in_features) * rho_prior
            bias_rho_init = torch.ones(out_features) * rho_prior
        # initialise to prior
        else:
            weights_mu_init = weights_mu_prior
            bias_mu_init = bias_mu_prior
            weights_rho_init = weights_rho_prior
            bias_rho_init = bias_rho_prior

        if prior_dist == "gaussian":
            dist = Gaussian
        elif prior_dist == "laplace":
            dist = Laplace
        else:
            raise RuntimeError(f"Wrong prior_dist {prior_dist}")

        self.bias = dist(
            bias_mu_init.clone(),
            bias_rho_init.clone(),
            device=device,
            fix_mu=fix_mu,
            fix_rho=fix_rho,
        )
        self.weight = dist(
            weights_mu_init.clone(),
            weights_rho_init.clone(),
            device=device,
            fix_mu=fix_mu,
            fix_rho=fix_rho,
        )
        self.weight_prior = dist(
            weights_mu_prior.clone(),
            weights_rho_prior.clone(),
            device=device,
            fix_mu=True,
            fix_rho=True,
        )
        self.bias_prior = dist(
            bias_mu_prior.clone(),
            bias_rho_prior.clone(),
            device=device,
            fix_mu=True,
            fix_rho=True,
        )

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior
            ) + self.bias.compute_kl(self.bias_prior)

        return F.linear(input, weight, bias)

    def compute_kl_force(self):
        self.kl_div = self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(
            self.bias_prior
        )

    def __deepcopy__(self, memo):
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        self.kl_div = self.kl_div.detach().clone()
        cp = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method
        cp.__deepcopy__ = deepcopy_method
        return cp

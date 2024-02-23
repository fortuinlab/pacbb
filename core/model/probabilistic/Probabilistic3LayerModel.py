import torch
import torch.nn as nn


class Probabilistic3LayerModel(nn.Module):
    """Implementation of a Probabilistic Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : NNet object
        Network object used to initialise the prior

    """

    def __init__(
        self,
        rho_prior,
        prior_dist="gaussian",
        device="cuda",
        init_net=None,
        features=28 * 28,
        classes=10,
        neurons=100,
        init_prior="weights",
        init_prior_net=None,
        fix_mu=False,
        fix_rho=False,
    ):
        super().__init__()
        self.l1 = ProbLinear(
            features,
            neurons,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.l1 if init_net else None,
            init_prior=init_prior,
            fix_mu=fix_mu,
            fix_rho=fix_rho,
        )

        self.l2 = ProbLinear(
            neurons,
            neurons,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.l2 if init_net else None,
            init_prior=init_prior,
            fix_mu=fix_mu,
            fix_rho=fix_rho,
        )
        self.l3 = ProbLinear(
            neurons,
            classes,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.l3 if init_net else None,
            init_prior=init_prior,
            fix_mu=fix_mu,
            fix_rho=fix_rho,
        )
        self.features = features
        self.classes = classes

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        x = x.view(-1, self.features)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = output_transform(self.l3(x, sample), clamping, pmin)
        return x

    def compute_kl(self, recompute=False):
        # KL as a sum of the KL for each individual layer
        if recompute:
            return self.l1.compute_kl() + self.l2.compute_kl() + self.l3.compute_kl()
        else:
            return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div

    def compute_kl_force(self):
        self.l1.compute_kl_force()
        self.l2.compute_kl_force()
        self.l3.compute_kl_force()
        return self.compute_kl()

    def _replace_weights_and_biases(
        self, layer, device, dist, la_posterior_precision, shift
    ):
        weights_mu_prior = layer.weight
        bias_mu_prior = layer.bias
        weights_rho_prior = la_posterior_precision[
            shift : shift + layer.out_features * layer.in_features
        ].reshape(layer.out_features, layer.in_features)
        bias_rho_prior = la_posterior_precision[
            shift
            + layer.out_features * layer.in_features : shift
            + layer.out_features * layer.in_features
            + layer.out_features
        ]
        bias = dist(
            bias_mu_prior.clone(),
            bias_rho_prior.clone(),
            device=device,
            fix_mu=False,
            fix_rho=False,
        )
        weight = dist(
            weights_mu_prior.clone(),
            weights_rho_prior.clone(),
            device=device,
            fix_mu=False,
            fix_rho=False,
        )
        return bias, weight

    def replace_weights_and_biases(
        self, net, device, la_posterior_precision, dist_name
    ):
        shift = 0
        if dist_name == "gaussian":
            dist = Gaussian
        elif dist_name == "laplace":
            dist = Laplace
        else:
            raise RuntimeError(f"Wrong prior_dist {dist_name}")

        bias, weight = self._replace_weights_and_biases(
            layer=net.l1,
            device=device,
            dist=dist,
            la_posterior_precision=la_posterior_precision,
            shift=shift,
        )
        self.l1.bias = bias
        self.l1.weight = weight
        if net is not None:
            shift += net.l1.out_features * net.l1.in_features + net.l1.out_features
        bias, weight = self._replace_weights_and_biases(
            layer=net.l2,
            device=device,
            dist=dist,
            la_posterior_precision=la_posterior_precision,
            shift=shift,
        )
        self.l2.bias = bias
        self.l2.weight = weight
        if net is not None:
            shift += net.l2.out_features * net.l2.in_features + net.l2.out_features
        bias, weight = self._replace_weights_and_biases(
            layer=net.l3,
            device=device,
            dist=dist,
            la_posterior_precision=la_posterior_precision,
            shift=shift,
        )
        self.l3.bias = bias
        self.l3.weight = weight
        if net is not None:
            shift += net.l3.out_features * net.l3.in_features + net.l3.out_features

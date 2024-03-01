import pytest
import torch

from core.trainer import TorchOptimizerFactory


@pytest.mark.parametrize(
    "optimizer_name, parameters",
    [("sgd", {"lr": 0.001, "momentum": 0.9})],
)
def test_torch_objective(optimizer_name, parameters, pbp3_model):
    parameters["params"] = pbp3_model.parameters()
    optimizer = TorchOptimizerFactory().create(optimizer_name, **parameters)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.ones((28, 28))
    y_true = torch.randn((1, 10))

    pbp3_model.train()

    y_pred = pbp3_model.forward(x)
    loss = criterion(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import pytest
import torch
import torch.nn.functional as F

from core.trainer.objective import ObjectiveFactory


@pytest.mark.parametrize(
    "objective_name, parameters",
    [
        ('bbb', {'kl_penalty': 0.001, 'pmin': 1e-4, 'num_classes': 10, 'device': torch.device('cpu')})
    ],
)
def test_torch_objective(objective_name, parameters, pbp3_model):
    objective = ObjectiveFactory().create(objective_name, **parameters)
    criterion = torch.nn.NLLLoss()

    x = torch.ones((28, 28))
    target = torch.Tensor([5])
    # y_true = torch.zeros(parameters['num_classes'])
    # y_true[int(target.item()) - 1] = 1
    # y_true = y_true.view(1, -1)
    torch.manual_seed(42)
    y_pred = pbp3_model.forward(x)
    loss_ce = criterion(y_pred, target.long())
    kl = pbp3_model.compute_kl()
    bound = objective.bound(loss_ce, kl, 1)
    assert bound >= 0

    torch.manual_seed(42)
    bound_copy, kl_copy, y_pred_copy, loss_ce_copy, loss_01 = objective.train_objective(model=pbp3_model,
                                                                                        data=x,
                                                                                        target=target.long(),
                                                                                        num_samples=1)
    assert bound == bound_copy
    assert kl == kl_copy
    assert torch.eq(y_pred, y_pred_copy).all()
    assert loss_ce == loss_ce_copy

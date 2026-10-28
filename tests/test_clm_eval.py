import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
import finetune  # noqa: E402


def test_val_top1_ignores_same_step_actions_masked_out_of_the_loss():
    # Rows 0 and 1 are two rollouts of one task at the same step with identical
    # state and action; the loss does not treat them as negatives for each other.
    torch.manual_seed(0)
    states = torch.randn(3, 8)
    states[1] = states[0]
    actions = states.clone()
    codes = torch.tensor([0, 0, 1])
    data = (states, actions, codes, torch.tensor([0, 0, 0]), None, 1)
    identity = torch.nn.Identity()
    metrics = finetune._evaluate_clm(identity, identity, torch.nn.Parameter(torch.tensor(0.0)), data, 4)
    assert metrics["within_task_top1"] == 1.0
    assert metrics["val_top1"] == 1.0

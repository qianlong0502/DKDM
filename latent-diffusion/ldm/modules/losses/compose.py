from re import S
from ldm.modules.losses.frequency_loss import SpecAwareLoss
from torch import nn

class ComposeLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        assert len(losses) == len(weights), 'Number of loss must be the same as weights'
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        self.number_loss = len(weights)

    def forward(self, pred, target, ref=None):
        loss_value = 0.0
        for i in range(self.number_loss):
            if isinstance(self.losses[i], SpecAwareLoss):
                assert ref is not None
                loss_value += self.losses[i](pred, target, ref).mean() * self.weights[i]
            else:
                loss_value += self.losses[i](pred, target).mean() * self.weights[i]
        return loss_value
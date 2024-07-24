import torch.nn as nn
from torch.nn import MSELoss, L1Loss

class Losses(nn.Module):
    def __init__(self, types, weights):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.types = types
        self.weights = weights
        for loss_type in types:
            if loss_type == 'MSE':
                self.module_list.append(MSELoss())
            elif loss_type == 'L1':
                self.module_list.append(L1Loss())

    def __len__(self):
        return len(self.types)

    def forward(self, preds, gts):
        losses = []
        for i in range(len(self.types)):
            loss = self.module_list[i](preds[i],gts[i]) * self.weights[i]
            losses.append(loss)
        return losses

def build_loss(config):
    loss_types = config['types']
    loss_weights = config['weights']

    assert len(loss_weights) == len(loss_types)
    criterion = Losses(types=loss_types, weights=loss_weights)
    return criterion

import torch
import torch.nn as nn

class CustomWeightedLoss(nn.Module):
    def __init__(self, weight1, weight2):
        super(CustomWeightedLoss, self).__init__()
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, loss1, loss2):
        weighted_loss = self.weight1 * loss1 + self.weight2 * loss2
        return weighted_loss
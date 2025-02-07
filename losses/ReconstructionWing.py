import math

import numpy as np
import torch
import wandb
from sklearn.tree._criterion import MSE
from tslearn.metrics import SoftDTWLossPyTorch

from utils.utils_dl import DLUtils


class EEGReconstructionLoss(torch.nn.Module):
    def __init__(self, device, omega=10, epsilon=2):
        super(EEGReconstructionLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

        self.DTW = SoftDTWLossPyTorch(gamma=1.0)
        self.MSE = torch.nn.MSELoss()

        self.device = device

    def forward(self, x_in, x_rec):
        recmse, recdtwdist, l_connectivity = 0, 0, 0

        n = len(x_in)
        for i in range(n):
            x_in_0, x_dec_0 = torch.unsqueeze(x_in[i], dim=0), torch.unsqueeze(x_rec[i], dim=0)
            x_in_0, x_dec_0 = x_in_0.to(self.device), x_dec_0.to(self.device)

            '''Structure'''
            recmse += self.applyWingConsideration(self.MSE(x_in_0, x_dec_0), self.omega, self.epsilon)
            dist = self.applyWingConsideration(self.DTW(x_in_0, x_dec_0), self.omega, self.epsilon)

            recdtwdist += torch.mean(dist)

            '''Connectivity'''
            l_connectivity += self.computeConnectivityLosses(x_in_0[0].T, x_dec_0[0].T)

        recmse = recmse / n
        recdtwdist = recdtwdist / n
        loss = recmse + 0.000001*recdtwdist + 0.1*l_connectivity
        wandb.log({"recmse_train": recmse, "recdtw_train": 0.000001*recdtwdist, "connectivity_train": 0.01*l_connectivity,
                    "loss_train": loss})

        return loss


    def computeConnectivityLosses(self, x, y):
        taumax = 10
        x_cau = DLUtils.computeCorrPCMI(DLUtils.genTigraDataFrame(x, gfp=True), taumax)
        y_cau = DLUtils.computeCorrPCMI(DLUtils.genTigraDataFrame(y, gfp=True), taumax)

        tx_cau, ty_cau = torch.tensor(x_cau), torch.tensor(y_cau)
        loss_frobenius = self.applyWingConsideration(torch.norm(tx_cau - ty_cau, p='fro'), self.omega, self.epsilon)

        loss_jaccard = self.applyWingConsideration(self.computeJaccardSimilarity(tx_cau, ty_cau), self.omega, self.epsilon)
        loss_jaccard = torch.tensor(loss_jaccard)

        COSSIM = torch.nn.CosineSimilarity()
        loss_cosine = 2 - (torch.mean(COSSIM(tx_cau, ty_cau)) + 1)
        loss_cosine = self.applyWingConsideration(loss_cosine, self.omega, self.epsilon)

        loss = loss_frobenius + loss_jaccard + loss_cosine

        return loss

    def computeJaccardSimilarity(self, Ax, Ay):
        flat_adj1 = Ax.flatten()
        flat_adj2 = Ay.flatten()

        # Convert to binary (if they are not already)
        flat_adj1 = torch.where(flat_adj1 > 0, 1, 0)
        flat_adj2 = torch.where(flat_adj2 > 0, 1, 0)

        # Compute the intersection and union
        intersection = torch.sum(torch.logical_and(flat_adj1, flat_adj2))
        union = torch.sum(torch.logical_or(flat_adj1, flat_adj2))

        # Compute Jaccard similarity
        jaccard_sim = intersection / union if union != 0 else 1.0
        jaccard_sim = 1 - jaccard_sim

        return jaccard_sim
    def applyWingConsideration(self, delta_y, omega, epsilon):
        delta_y1 = delta_y[delta_y < omega]
        delta_y2 = delta_y[delta_y >= omega]
        loss1 = omega * torch.log(1 + delta_y1 / epsilon)
        C = omega - omega * math.log(1 + omega / epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
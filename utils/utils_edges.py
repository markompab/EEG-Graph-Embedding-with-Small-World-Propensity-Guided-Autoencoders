import numpy as np
import torch
from mne_connectivity import phase_slope_index


class UtilsEdges:
    @staticmethod
    def edge2Adj(edges, edgeweights, n):

        adj = np.zeros((n, n))
        for i in range(edges.shape[1]):
            adj[edges[0, i], edges[1, i]] = edgeweights[i]
            adj[edges[1, i], edges[0, i]] = edgeweights[i]

        return adj

    @staticmethod
    def edges2SquareMat(edges, edgeweights,  sqlen):

        matrix = np.zeros((sqlen, sqlen))
        n = edgeweights.shape[0]

        for i in range(n):
            r, c = edges[0][i], edges[1][i]
            matrix[r, c] = edgeweights[i]
            matrix[c, r] = edgeweights[i]

        return matrix

    @staticmethod
    def computePSI(raweeg, band="all"):
        freq_bands = {
            "delta": [0, 4]
            , "theta": [4, 7]
            , "alpha": [8, 12]
            , "beta": [12, 30]
            , "gamma": [30, 45]
            , "all": [0, 45]
        }

        frng = freq_bands[band]
        fmin, fmax, tmin_con, sfreq = frng[0], frng[1], 0., 512
        psi_comb_indices = ([], [])

        for i in range(19):
            for j in range(i + 1, 19):
                psi_comb_indices[0].append(i)
                psi_comb_indices[1].append(j)

        raweeg = np.expand_dims(raweeg, axis=0)

        psi = phase_slope_index(raweeg, mode='multitaper', indices=psi_comb_indices, sfreq=sfreq, fmin=fmin, fmax=fmax, tmin=tmin_con)

        psi_sq = UtilsEdges.edges2SquareMat(psi_comb_indices, psi.get_data(), 19)

        return torch.Tensor(psi_sq)
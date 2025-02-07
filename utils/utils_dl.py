from multiprocessing import Pool

import networkx as nx
from itertools import combinations

import mne.filter
import numpy as np
import pandas
import pandas as pd
import scipy
import torch
from mne_connectivity import phase_slope_index
from sklearn import preprocessing

from tigramite.independence_tests.parcorr import ParCorr
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCIbase, PCMCI

from customlibs.ctigramitetorch.data_processing import DataFrameTorch
from customlibs.ctigramitetorch.independence_tests.parcorr import ParCorrTorch
from customlibs.ctigramitetorch.pcmci import PCMCITorch


class DLUtils:
    @staticmethod
    def genEdges(ndcnt):
        edges = []
        for i in range(ndcnt):
            for j in range(i, ndcnt):
                if (i == j):
                    continue
                edges.append((i, j))

        edges = np.array(edges, dtype=np.int64)
        edges = np.swapaxes(edges, 0, 1)
        edges = torch.Tensor(edges).type(torch.int64)

        return edges

    @staticmethod
    def genEdges361(ndcnt):
        edges = []
        for i in range(ndcnt):
            for j in range(0, ndcnt):
                if (i == j):
                    continue
                edges.append((i, j))

        edges = np.array(edges, dtype=np.int64)
        edges = np.swapaxes(edges, 0, 1)
        edges = torch.Tensor(edges).type(torch.int64)

        return edges

    @staticmethod
    def genEdgesPSI(raweeg, bands="all",  usecorrweights=True):

        freq_bands = {
            "delta": [0, 4]
            , "theta": [4, 7]
            , "alpha": [8, 12]
            , "beta": [12, 30]
            , "gamma": [30, 45]
            , "all": [0, 45]
            #, "all": [0, 100]
        }

        frng = freq_bands[bands]
        fmin, fmax, tmin_con, sfreq = frng[0], frng[1], 0., 512
        psi_comb_indices = ([], [])
        n = raweeg.shape[0]

        print("n:{}".format(n))
        if(n==1):

            psi_comb_indices[0].append(0)
            psi_comb_indices[1].append(0)
            psi_vals = np.array([1.0])

            return torch.tensor(psi_comb_indices), torch.tensor(psi_vals)
        

        for i in range(n):
            for j in range(i + 1, n):
                psi_comb_indices[0].append(i)
                psi_comb_indices[1].append(j)

        #raweeg = pandas.read_csv(srcpath).values.T
        raweeg = np.expand_dims(raweeg, axis=0)

        psi = phase_slope_index(raweeg, mode='multitaper', indices = psi_comb_indices, sfreq=sfreq, fmin=fmin, fmax=fmax, tmin=tmin_con, verbose=False)
        psi_vals  = psi.get_data()

        if(psi_vals.shape[1]>1):
            psi_vals  = DLUtils.normalizeNp(psi_vals)

        return torch.tensor(psi_comb_indices), torch.tensor(psi_vals)

    @staticmethod
    def genEdgesPSICRange(raweeg, bands, usecorrweights=True):

        freq_bands = {
            "delta": [0, 4]
            , "theta": [4, 7]
            , "alpha": [8, 12]
            , "beta": [12, 30]
            , "gamma": [30, 45]
            , "all": [0, 45]
        }

        # frng = freq_bands[bands]
        fmin, fmax, tmin_con, sfreq = bands[0], bands[1], 0., 500
        psi_comb_indices = ([], [])
        n = raweeg.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                psi_comb_indices[0].append(i)
                psi_comb_indices[1].append(j)

        # raweeg = pandas.read_csv(srcpath).values.T
        raweeg = np.expand_dims(raweeg, axis=0)

        psi = phase_slope_index(raweeg, mode='multitaper', indices=psi_comb_indices, sfreq=sfreq, fmin=fmin, fmax=fmax,
                                tmin=tmin_con)
        psi_vals = psi.get_data()
        psi_vals = DLUtils.normalizeNp(psi_vals)
        return torch.tensor(psi_comb_indices), torch.tensor(psi_vals)

    @staticmethod
    def genEdgesPLV(raweeg):

        """
        Compute the Phase Locking Values (PLVs) for all pairs of channels.

        Parameters:
        - data: 2D array-like, shape (n_channels, n_samples)
        - n_channels: int, the number of channels

        Returns:
        - plvs: dict, keys are pairs of channel indices and values are PLVs
        """
        nchannels = raweeg.shape[0]
        # Generate all unique pairs of channel indices
        channel_pairs = list(combinations(range(nchannels), 2))

        # Prepare data for each pair
        pair_data = [(raweeg[ch1], raweeg[ch2]) for ch1, ch2 in channel_pairs]

        # Use multiprocessing to compute PLVs in parallel
        plvs = []
        for pdrow in pair_data:
            plvs.append(DLUtils.compute_plv_pair(pdrow))

        tedges = torch.tensor(channel_pairs)
        tweights = torch.tensor(plvs)

        tedges =  tedges.T
        tweights =  torch.unsqueeze(tweights, 0).T
        return tedges, tweights
    
    @staticmethod
    def genEdgesPCMCI(x_in):
        
        df_tigra = DLUtils.genTigraDataFrame(x_in.T, gfp=False)
        pcau = DLUtils.computeCorrPCMI(df_tigra, 2)[:, :, 0]
        edges, edge_attr = DLUtils.causality2EdgeCorr(torch.Tensor(pcau))
        
        return torch.tensor(edges), torch.tensor(edge_attr)
    
    @staticmethod
    def compute_plv_pair(pair_data):
        signal1, signal2 = pair_data
        """
        Compute the Phase Locking Value (PLV) between two signals.

        Parameters:
        - signal1: 1D array-like, first signal
        - signal2: 1D array-like, second signal

        Returns:
        - plv: float, the Phase Locking Value between the two signals
        """
        # Compute the analytic signal using the Hilbert transform
        analytic_signal1 = scipy.signal.hilbert(signal1)
        analytic_signal2 = scipy.signal. hilbert(signal2)

        # Obtain the instantaneous phase
        phase1 = np.angle(analytic_signal1)
        phase2 = np.angle(analytic_signal2)

        # Calculate the phase difference
        phase_difference = phase1 - phase2

        # Compute the PLV
        plv = np.abs(np.mean(np.exp(1j * phase_difference)))

        return plv

    @staticmethod
    def genEdgesPSIKids(raweeg, bands="all",  usecorrweights=True):

        freq_bands = {
            "delta": [1, 4]
            , "theta": [4, 8]
            , "alpha": [8, 13]
            , "beta": [13, 30]
            , "gamma": [30, 100]
            , "all": [0, 45]
        }

        frng = freq_bands[bands]
        fmin, fmax, tmin_con, sfreq = frng[0], frng[1], 0., 512
        psi_comb_indices = ([], [])
        n = raweeg.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                psi_comb_indices[0].append(i)
                psi_comb_indices[1].append(j)

        #raweeg = pandas.read_csv(srcpath).values.T
        raweeg = np.expand_dims(raweeg, axis=0)

        psi = phase_slope_index(raweeg, mode='multitaper', indices=psi_comb_indices, sfreq=sfreq, fmin=fmin, fmax=fmax, tmin=tmin_con)
        psi_vals  = psi.get_data()

        return torch.tensor(psi_comb_indices), torch.tensor(psi_vals)

    @staticmethod
    def genEdgesPSIRob(raweeg, bands="all",  usecorrweights=True):

        freq_bands = {
            "delta": [0, 4]
            , "theta": [4, 7]
            , "alpha": [8, 12]
            , "beta": [12, 30]
            , "gamma": [30, 45]
            , "all": [0, 45]
        }

        frng = freq_bands[bands]
        fmin, fmax, tmin_con, sfreq = frng[0], frng[1], 0., 512
        psi_comb_indices = ([], [])
        n = raweeg.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                psi_comb_indices[0].append(i)
                psi_comb_indices[1].append(j)

        #raweeg = pandas.read_csv(srcpath).values.T
        raweeg = np.expand_dims(raweeg, axis=0)

        psi = phase_slope_index(raweeg, mode='multitaper', indices=psi_comb_indices, sfreq=sfreq, fmin=fmin, fmax=fmax, tmin=tmin_con)
        psi_vals  = psi.get_data()

        edges, attr = [], []
        #m = psi_vals.shape[0]
        nrows, ncols = DLUtils.get_ij(psi_vals.shape[0], n)
        for i in range(nrows):
            for j in range(i, n):
                index = i * ncols + j
                if (psi_vals[index] < 0.3):
                    continue
                edges.append((i, j))

                if (usecorrweights):
                    attr.append(psi_vals[index])
                else:
                    attr.append(1)
        edges = torch.tensor(edges).T.int()
        return edges, torch.tensor(attr)

    @staticmethod
    def genEdgesCorr(dt, usecorrweights=True):
        #dt1 = dt.to_numpy().T
        corr = np.corrcoef(dt)
        edges, attr = [], []
        nd = corr.shape[0]

        for i in range(nd):
            for j in range(i, nd):
                if (corr[i, j] > 0.3):
                    edges.append((i, j))

                    if(usecorrweights):
                        attr.append(corr[i, j])
                    else:
                        attr.append(1)

        #edges = np.array(edges, dtype=np.int64).T
        #attr = np.array(attr, dtype=np.float32)

        edges = torch.Tensor(edges).T.int()
        attr  = torch.Tensor(attr)

        return edges, attr

    @staticmethod
    def genEdgesCorr4GFP(corr, usecorrweights=True):
        edges, attr = [], []
        nd = corr.shape[0]

        for i in range(nd):
            for j in range(i, nd):
                if (corr[i, j] > 0.3):
                    edges.append((i, j))

                    if (usecorrweights):
                        attr.append(corr[i, j])
                    else:
                        attr.append(1)

        edges = torch.Tensor(edges).T.int()
        attr = torch.Tensor(attr)

        return edges, attr

    @staticmethod
    def genEdgesCorrBands(raweeg, bands="all",  usecorrweights=True):

        freq_bands = {
            "delta": [0, 4]
            , "theta": [4, 7]
            , "alpha": [8, 12]
            , "beta": [12, 30]
            , "gamma": [30, 45]
            , "all": [0, 45]
        }

        frng = freq_bands[bands]
        fmin, fmax, tmin_con, sfreq = frng[0], frng[1], 0., 512
        eegf =  mne.filter.filter_data(raweeg, sfreq, fmin, fmax, verbose=False)
        corr = np.corrcoef(eegf)

        n = raweeg.shape[0]
        edges, attr = [], []
        nd = corr.shape[0]

        for i in range(nd):
            for j in range(i, nd):
                if (corr[i, j] > 0.3):
                    edges.append((i, j))

                    if(usecorrweights):
                        attr.append(corr[i, j])
                    else:
                        attr.append(1)

        edges = torch.Tensor(edges).T.int()
        attr  = torch.Tensor(attr)

        return edges, attr
    @staticmethod
    def genEdgesCovDst(dt, usecorrweights=True):

        sraweeg = dt.iloc[:8500, :]
        nraweeg = DLUtils.normalize(sraweeg)

        #dt1 = dt.to_numpy().T
        corr = np.corrcoef(dt)
        edges, attr = [], []
        nd = len(dt.columns)
        grid  = DLUtils.genGrid(dt.columns)
        edgesq = DLUtils.genEdgeList(nd)
        dists = DLUtils.getDistances(edgesq, grid)

        n = len(grid)
        dstweight = DLUtils.getDstWeights(dists)
        dstweight_sq = np.reshape(dstweight, [n, n])

        pcvar = dt.cov().values  # [:, :maxlen]
        pcvar_w = pcvar * dstweight_sq

        for i in range(nd):
            for j in range(i, nd):
                if (pcvar_w[i, j] > 0.5):
                    edges.append((i, j))

                    if(usecorrweights):
                        attr.append(corr[i, j])
                    else:
                        attr.append(1)

        #edges = np.array(edges, dtype=np.int64).T
        #attr = np.array(attr, dtype=np.float32)

        edges = torch.Tensor(edges).T.int()
        attr  = torch.Tensor(attr)

        return edges, attr

    @staticmethod
    def genEdgesCorr1(dt, usecorrweights=True):
        dt1 = dt.to_numpy().T
        corr = np.corrcoef(dt1)
        edges, attr = [], []
        nd = corr.shape[0]

        for i in range(nd):
            for j in range(i, nd):
                if (corr[i, j] >= 0.5):
                    edges.append((i, j))

                    if(usecorrweights):
                        attr.append(corr[i, j])
                    else:
                        attr.append(1)

        #edges = np.array(edges, dtype=np.int64).T
        #attr = np.array(attr, dtype=np.float32)

        edges = np.array(edges, dtype=np.int32).T
        attr  = np.array(attr, dtype=np.float32)

        return (edges, attr)


    @staticmethod
    def genEdgesCorr2(dt, usecorrweights=True):

        corr = np.corrcoef(dt)
        edges, attr = [], []
        nd = corr.shape[0]

        for i in range(nd):
            for j in range(i, nd):
                if (corr[i, j] >= 0.5):
                    edges.append((i, j))

                    if(usecorrweights):
                        attr.append(corr[i, j])
                    else:
                        attr.append(1)

        #edges = np.array(edges, dtype=np.int64).T
        #attr = np.array(attr, dtype=np.float32)

        edges = np.array(edges, dtype=np.int32).T
        attr  = np.array(attr, dtype=np.float32)

        return edges, attr

    @staticmethod
    def causality2EdgeCorr0(corr, usecorrweights=True):

        edges, attr = [], []
        nd = corr.shape[0]

        for i in range(nd):
            for j in range(i, nd):
                # if (corr[i, j] >= 0.5):
                edges.append((i, j))

                if (usecorrweights):
                    attr.append(corr[i, j])
                else:
                    attr.append(1)

        # edges = np.array(edges, dtype=np.int64).T
        # attr = np.array(attr, dtype=np.float32)

        edges = torch.Tensor(edges).T.type(torch.int64)
        attr = torch.Tensor(attr).type(torch.float32)

        return edges, attr

    @staticmethod
    def causality2EdgeCorr(corr, usecorrweights=True):

        edges, attr = [], []
        nd = corr.shape[0]

        for i in range(nd):
            for j in range(i, nd):
                if (corr[i, j] < 0.1):
                    continue

                edges.append((i, j))

                if (usecorrweights):
                    attr.append(corr[i, j])
                else:
                    attr.append(1)

        # edges = np.array(edges, dtype=np.int64).T
        # attr = np.array(attr, dtype=np.float32)

        edges = torch.Tensor(edges).T.type(torch.int64)
        attr  = torch.Tensor(attr).type(torch.float32)

        return edges, attr

    @staticmethod
    def causality2EdgeCorrNp(corr, usecorrweights=True):

        edges, attr = [], []
        nd = corr.shape[0]

        for i in range(nd):
            for j in range(i, nd):
                if (corr[i, j] < 0.1):
                    continue

                edges.append((i, j))

                if (usecorrweights):
                    attr.append(corr[i, j])
                else:
                    attr.append(1)

        edges, attr = np.array(edges), np.array(attr)
        return edges, attr

    @staticmethod
    def causAll2EdgeCorr(corr, usecorrweights=True):

        edges, attr = [], []
        nd = corr.shape[0]

        for i in range(nd):
            for j in range(i, nd):
                #if (corr[i, j] >= 0.5):
                edges.append((i, j))

                if(usecorrweights):
                    attr.append(corr[i, j, :])
                else:
                    attr.append(1)

        #edges = np.array(edges, dtype=np.int64).T
        #attr = np.array(attr, dtype=np.float32)

        edges = torch.Tensor(edges).T.type(torch.int64)
        attr  = torch.Tensor(attr).type(torch.float32)

        return edges, attr


    @staticmethod
    def getDistances(edges, grid):
        distances = []
        for i in range(len(edges)):
            dst = np.linalg.norm(grid[edges[i][0]] - grid[edges[i][1]])
            distances.append(dst)

        return distances


    @staticmethod
    def getDstWeights(distances):
        ''''''
        n = len(distances)
        max = np.max(distances)
        weights = []
        for i in range(n):
            weight = 1 - (distances[i] / max)
            weights.append(weight)

        return weights

    @staticmethod
    def genGrid(cols):
        pos = {
            "Fp2": [-0.30699, 0.94483]  # Fp2
            , "Fp1": [0.30699, 0.94483]  # Fp1

            , "F8": [-0.80377, 0.58397]  # F8
            , "F4": [-0.56611, 0.69909]  # F4
            , "Fz": [0, 0.76041]  # Fz
            , "F3": [0.56611, 0.69909]  # F3
            , "F7": [0.80377, 0.58397]  # F7

            , "T8": [-0.99361, 0]  # T8
            , "C4": [-0.75927, 0.00133]  # C4
            , "Cz": [0, 0.00175]  # Cz
            , "C3": [0.75927, 0.00133]  # C3
            , "T7": [0.99361, 0]  # T7

            , "P8": [-0.80393, -0.58409]  # P8
            , "P4": [-0.56563, -0.69849]  # P4
            , "Pz": [0, -0.75813]  # Pz
            , "P3": [0.56563, -0.69849]  # P3
            , "P7": [0.80393, -0.58409]  # P7

            , "O2": [-0.30709, -0.94513]  # O2
            , "O1": [0.30709, -0.94513]  # O1
        }

        coords = []

        for key in cols:
            coords.append(pos[key])

        coords = np.array(coords)

        return coords

    @staticmethod
    def getElectrodeGrid():
        pos = [
            [-0.30699, 0.94483]  # Fp2
            , [0.30699, 0.94483]  # Fp1

            , [-0.80377, 0.58397]  # F8
            , [-0.56611, 0.69909]  # F4
            , [0, 0.76041]  # Fz
            , [0.56611, 0.69909]  # F3
            , [0.80377, 0.58397]  # F7

            , [-0.99361, 0]  # T8
            , [-0.75927, 0.00133]  # C4
            , [0, 0.00175]  # Cz
            , [0.75927, 0.00133]  # C3
            , [0.99361, 0]  # T7

            , [-0.80393, -0.58409]  # P8
            , [-0.56563, -0.69849]  # P4
            , [0, -0.75813]  # Pz
            , [0.56563, -0.69849]  # P3
            , [0.80393, -0.58409]  # P7

            , [-0.30709, -0.94513]  # O2
            , [0.30709, -0.94513]  # O1
        ]
        pos = torch.Tensor(pos)
        return pos

    @staticmethod
    def genEdgeList(n):
        edges = []

        for i in range(n):
            for j in range(n):
                edges.append([i,j])

        return edges

    @staticmethod
    def normalize(df):
        x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        return df

    @staticmethod
    def genTigraDataFrame(data, gfp=False):
        var_names = ['Fp2', 'Fp1', 'F8', 'F4', 'Fz'
            , 'F3', 'F7', 'T8', 'C4', 'Cz'
            , 'C3', 'T7', 'P8', 'P4', 'Pz'
            , 'P3', 'P7', 'O2', 'O1']

        if(gfp):
            var_names.append("GFP")

        dataframe = pp.DataFrame(data.cpu().detach().numpy(),
                                 datatime={0: np.arange(len(data))},
                                 var_names=var_names)

        return dataframe

    @staticmethod
    def computeCorrPCMI(dfTigra, taumax=20):
        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(
            dataframe=dfTigra,
            cond_ind_test=parcorr,
            verbosity=1)

        pcmci.verbosity = 0
        # correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']
        correlations = pcmci.get_lagged_dependencies(tau_max=taumax, val_only=True)['val_matrix']

        # results = pcmci.run_pcmci(tau_max=3, pc_alpha=None, alpha_level=0.01)
        return correlations

    @staticmethod
    def genTigraDataFrameTorch(data, gfp=False):
        var_names = ['Fp2', 'Fp1', 'F8', 'F4', 'Fz'
            , 'F3', 'F7', 'T8', 'C4', 'Cz'
            , 'C3', 'T7', 'P8', 'P4', 'Pz'
            , 'P3', 'P7', 'O2', 'O1']

        if (gfp):
            var_names.append("GFP")

        # dataframe = CDataFrame(data.cpu().detach().numpy(),
        #                          datatime={0: torch.arange(len(data))},
        #                          var_names=var_names)
        dataframe = DataFrameTorch(data,
                                   datatime={0: torch.arange(len(data)).to(data.device)},
                                   var_names=var_names)

        return dataframe

    @staticmethod
    def computeCorrPCMITorch(dfTigra, taumax=20):
        parcorr = ParCorrTorch(significance='analytic')
        pcmci = PCMCITorch(
            dataframe=dfTigra,
            cond_ind_test=parcorr,
            verbosity=1)

        pcmci.verbosity = 0
        # correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']
        correlations = pcmci.get_lagged_dependencies(tau_max=taumax, val_only=True)['val_matrix']

        # results = pcmci.run_pcmci(tau_max=3, pc_alpha=None, alpha_level=0.01)
        return correlations

    @staticmethod
    def get_ij(index, rowlen):
        i = int(index / rowlen)
        j = index % rowlen
        return i, j


    @staticmethod
    def normalizeNp(x):
        min = np.min(x)
        max = np.max(x)
        x_norm = (x - min) / (max - min)
        return x_norm


    @staticmethod
    def genNetxGraph(nodes, edges, edge_weight, pos):

        G = nx.Graph()
        sh = edges.shape

        for i in range(len(nodes)):
            G.add_node(nodes[i], pos=pos[i])

        for i in range(sh[0]):
            # if(edges[i][0] == edges[i][1]):
            #    continue
            w = edge_weight[i]
            if (w < 0.3):
                continue

            G.add_edge(edges[i][0], edges[i][1], weight=edge_weight[i])

        return G

    @staticmethod
    def toBatches(x, ndCnt):
        n = int(x.shape[0] / ndCnt)
        batches = []
        for i in range(n):
            batch = x[i * ndCnt:(i + 1) * ndCnt, :]
            batches.append(batch)

        return torch.stack(batches)


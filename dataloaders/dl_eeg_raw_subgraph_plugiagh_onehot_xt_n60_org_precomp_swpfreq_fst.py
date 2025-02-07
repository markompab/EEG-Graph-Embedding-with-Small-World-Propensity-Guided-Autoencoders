import traceback

import math

import os

import networkx as nx
import numpy as np
import pandas
import pandas as pd
import scipy
import torch
from torch_geometric.data import Data, Dataset

from utils.utils_clustering import UtilsClustering
from utils.utils_dl import DLUtils
from utils.utils_lst import UtilsLst
from utils.utils_tensor import UtilsTensor


class DLSubGPlgGAHNorm60OrgPSIAllSWPFreqPreFst(Dataset):


    def __init__(self, srcdir,  srcdircorrpre, srcdirsswpfreqranges,  srccsv, subggraph_nodes, column="iq", maxseqlen=8500, k=5, band="all", swnlen=5):
        super().__init__(srcdir)

        self.srcdir = srcdir
        self.srccsv = srccsv

        #self.data, self.slices = torch.load(self.processed_paths[0]
        #self.data
        files = os.listdir(srcdir)
        files = [f.replace(".csv", "") for f in files]

        self.dframe = pd.read_csv(srccsv)
        self.dframe = self.dframe[self.dframe["serial"].isin(files)]
        self.dframe = self.dframe[self.dframe[column] > 0 ]

        self.iqlabels = self.dframe[column].values
        self.ids      = self.dframe["serial"].values

        self.ages     = self.dframe["birth_months"].values
        self.genders  = self.dframe["gender"].values
        self.handed   = self.dframe["handed"].values
        self.maxage   = max(self.ages)

        self.k = k
        #
        # '''double'''
        # a, b = "_1", "_2"
        # seg_one = np.char.add(self.ids.tolist(), a)
        # seg_two = np.char.add(self.ids.tolist(), b)
        # self.ids     = np.hstack([seg_one, seg_two])
        # self.ages    = np.hstack([self.ages, self.ages])
        # self.genders = np.hstack([self.genders, self.genders])
        # self.handed  = np.hstack([self.handed, self.handed])
        # self.iqlabels = np.hstack([self.iqlabels, self.iqlabels])

        self.iqmax    = max(self.iqlabels)
        self.iqmin    = min(self.iqlabels)

        self.maxseqlen = maxseqlen
        self.items    = {}

        self.maxscore = 60

        self.subnodes = subggraph_nodes
        self.band= band

        self.swnlen = swnlen
        self.srcdircorrpre = srcdircorrpre
        self.srcdirsswpfreqranges = srcdirsswpfreqranges


    @property
    def raw_file_names(self):
        return self.ids

    @property
    def processed_file_names(self):
        return self.items

    def len(self):
        return len(self.ids)

    def get(self, idx):

        if(self.ids[idx] in list(self.items.keys())):
            return self.items[self.ids[idx]]

        srcpath = "{}/{}.csv".format(self.srcdir, self.ids[idx])

        # srcpath = "{}/{}.csv".format(self.srcdir, self.ids[idx][:-2])

        raweeg = pandas.read_csv(srcpath)
        raweeg = raweeg[self.subnodes]

        nodefeatures = torch.Tensor(raweeg.values)[:self.maxseqlen, :]
        nodefeatures = torch.nn.functional.normalize(nodefeatures, dim=1)
        nodefeatures = UtilsLst.replace_nan_and_inf(nodefeatures)
        nodefeatures = torch.swapaxes(nodefeatures, 0, 1)

        #edge_index, edge_attr = DLUtils.genEdgesPSI(nodefeatures, bands=self.band)
        srcpathswpfreqranges = "{}/{}.npz".format(self.srcdirsswpfreqranges, self.ids[idx])
        dt_swpfreq = np.load(srcpathswpfreqranges, allow_pickle=True)
        swp_freqrange = torch.tensor(dt_swpfreq["swp_freqrange"])

        srcpathmeta = "{}/{}.npz".format(self.srcdircorrpre, self.ids[idx])
        dt = np.load(srcpathmeta, allow_pickle=True)
        edge_index, edge_attr = torch.tensor(dt["edge_index"]), torch.tensor(dt["edge_attr"])

        iq = torch.tensor(self.iqlabels[idx], dtype=torch.float32).unsqueeze(0)
        target = torch.Tensor([[iq /self.maxscore]])

        age =torch.tensor(self.getAgeEncoding(self.ids[idx]), dtype=torch.float32)
        gender = torch.tensor(self.getGenderCode(self.genders[idx]), dtype=torch.float32)
        handed = torch.tensor(self.getHandedCode(self.handed[idx]), dtype=torch.float32)
        #klabels = self.getSpectralLabels(nodefeatures, self.k)
        #age = age/self.maxage

        swn_graph = self.getBstSWPGraph(edge_index, edge_attr, granularity=0.05)

        #swn_sm, swn_lg = self.getSmallWorldNetworks(edge_index, edge_attr, r=self.swnlen)

        data = Data(x=nodefeatures, edge_index=edge_index, edge_attr=edge_attr, y=target)
        #data = Data(x=nodefeatures, edge_index=edge_index, y=target)

        sample = {'graphdata': data
                ,  'iq': target
                , 'serial': self.ids[idx]
                , 'gender': gender
                , 'age': age
                , 'handed': handed
                , 'swn_graph': swn_graph
                , 'swp_freqrange': swp_freqrange

                #, 'klabels': klabels
            }

        self.items[self.ids[idx]] = sample

        #data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return sample

    def averageRefData(self, raweeg):
        avg =  torch.mean(raweeg, dim=1, keepdim=True)
        refeeg = raweeg - avg
        #refeeg = refeeg / torch.max(torch.abs(refeeg))
        return refeeg

    def getHandedCode(self, strHanded):

        encoding_handed = [0, 0, 0]

        if ("L" in strHanded):
            encoding_handed = [0, 0, 1]

        elif ("R" in strHanded):
            encoding_handed = [0, 1, 0]

        else:
            encoding_handed = [1, 0, 0]

        return encoding_handed

    def getGenderCode(self, strGender):

        encoding_gender = [0, 0]

        if ("M" in strGender):
            encoding_gender = [0, 1]

        else:
            encoding_gender = [1, 0]

        return encoding_gender

    def getSpectralLabels(self, raweeg, k=4):
        ''''''
        #xraweeg = np.unsqueeze(raweeg, axis=0)
        raweeg = torch.Tensor(raweeg)
        X =  UtilsClustering.genRepresentationCauSingle(raweeg, k)
        klabels = UtilsClustering.genClusters(X, k)
        klabels = torch.squeeze(klabels)

        return klabels



    def getBstSWPGraph(self, edges, edge_attr, granularity=0.05):
        ''''''
        #edges, edge_attr = edges.detach().numpy().T, edge_attr.detach().numpy()
        nds = torch.unique(edges)

        smn = torch.ones(19) * -1
        smn[:nds.shape[0]] = nds

        return smn


    def getAgeEncoding(self, strAge):

        encoding_age = 0

        if ("3-" in strAge):
            encoding_age = [0, 0, 1]

        elif ("4-" in strAge):
            encoding_age = [0, 1, 0]

        elif ("5-" in strAge):
            encoding_age = [1, 0, 0]

        return encoding_age

    def getLabel(self):
        lbl = "DLSubGPlgGAHNorm60OrgPSIAllSpectralX"
        return lbl


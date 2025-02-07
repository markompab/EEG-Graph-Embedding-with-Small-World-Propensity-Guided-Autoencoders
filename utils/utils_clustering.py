import numpy as np
import scipy as sp
import torch
from sklearn.cluster import KMeans

from utils.utils_dl import DLUtils
from utils.utils_edges import UtilsEdges
from utils.utils_tensor import UtilsTensor


class UtilsClustering:

        @staticmethod
        def get_clusters(data, n_clusters):
            """
            Get clusters from data
            :param data: Data
            :param n_clusters: Number of clusters
            :return: Clusters
            """
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(data)
            return kmeans.labels_


        @staticmethod
        def getDistWeightmap():
            grid = UtilsClustering.genGrid()
            edges = UtilsClustering.genEdgeList(19)
            dists = UtilsClustering.getDistances(edges, grid)

            dstweight = UtilsClustering.getDstWeights(dists)
            dstweight_sq = torch.reshape(dstweight, [19, 19])
            dstweight_sq = torch.nn.functional.normalize(dstweight_sq, p=1)

            return dstweight_sq


        @staticmethod
        def getDistances(edges, grid):
            distances = []
            for i in range(len(edges)):
                dst = torch.linalg.norm(grid[edges[i][0]] - grid[edges[i][1]])
                distances.append(dst)

            return distances

        @staticmethod
        def genEdgeList(n):
            edges = []

            for i in range(n):
                for j in range(n):
                    edges.append([i, j])

            return edges

        @staticmethod
        def getDstWeights(distances):
            ''''''
            n = len(distances)
            max = np.max(distances)
            weights = []
            for i in range(n):
                weight = 1 - (distances[i] / max)
                weights.append(weight)

            return torch.tensor(weights)

        @staticmethod
        def genGrid():
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
        def genRepresentation(x_in, k):

            '''cov'''

            #dstweight_sq = UtilsClustering.getDistWeightmap().to(x_in.device)
            dstweight_sq = UtilsClustering.getDistWeightmap()
            vsplit = torch.split(x_in, 19, 0)
            outComp = []

            n = int(x_in.shape[0] / 19)
            for i in range(n):
                pcov = torch.cov(vsplit[i])
                pcov = UtilsClustering.nonDiagNormalise(pcov)
                pcov_w = pcov * dstweight_sq

                plap = sp.sparse.csgraph.laplacian(pcov_w.to("cpu").numpy(), normed=True)
                plap = torch.Tensor(plap)

                w, v = torch.lobpcg(plap, k=k, largest=False)
                X = v * w

                # outComp.append(X)
                outComp.append(X)

            outComp = torch.cat(outComp, 0)

            return outComp

        @staticmethod
        def genRepresentationCorr(x_in, k):

            '''cov'''

            #dstweight_sq = UtilsClustering.getDistWeightmap().to(x_in.device)
            dstweight_sq = UtilsClustering.getDistWeightmap()
            vsplit = torch.split(x_in, 19, 0)
            outComp = []

            n = int(x_in.shape[0] / 19)
            for i in range(n):
                pcoeff = torch.corrcoef(vsplit[i])
                #pcoeff = UtilsClustering.nonDiagNormalise(pcoeff)
                pcov_w = pcoeff * dstweight_sq
                pcov_w = torch.nn.functional.normalize(pcov_w, p=1)

                # plap = sp.sparse.csgraph.laplacian(pcov_w.to("cpu").numpy(), normed=True)
                # plap = torch.Tensor(plap)
                plap = UtilsClustering.computeLaplacianMSCopilot(pcov_w)

                w, v = torch.lobpcg(plap, k=k, largest=False)
                X = v * w

                # outComp.append(X)
                outComp.append(X)

            outComp = torch.cat(outComp, 0)

            return outComp

        @staticmethod
        def genRepresentationCorrSingle(x_in, k):

            '''cov'''
            #dstweight_sq = UtilsClustering.getDistWeightmap().to(x_in.device)
            dstweight_sq = UtilsClustering.getDistWeightmap()
            #vsplit = torch.split(x_in, 19, 0)
            outComp = []

            n = int(x_in.shape[0] / 19)
            #for i in range(n):
            pcoeff = torch.corrcoef(x_in)
            #pcoeff = UtilsClustering.nonDiagNormalise(pcoeff)
            pcov_w = pcoeff * dstweight_sq
            pcov_w = torch.nn.functional.normalize(pcov_w, p=1)

            # plap = sp.sparse.csgraph.laplacian(pcov_w.to("cpu").numpy(), normed=True)
            # plap = torch.Tensor(plap)
            plap = UtilsClustering.computeLaplacianMSCopilot(pcov_w)

            w, v = torch.lobpcg(plap, k=k, largest=False)
            X = v * w

            # outComp.append(X)
            outComp.append(X)

            outComp = torch.cat(outComp, 0)

            return outComp


        @staticmethod
        def genRepresentationPSI0(x_in, k):

            '''cov'''

            #dstweight_sq = UtilsClustering.getDistWeightmap().to(x_in.device)
            dstweight_sq = UtilsClustering.getDistWeightmap()
            vsplit = torch.split(x_in, 19, 0)
            outComp = []

            n = int(x_in.shape[0] / 19)
            for i in range(n):
                pcoeff = torch.corrcoef(vsplit[i])
                #pcoeff = UtilsClustering.nonDiagNormalise(pcoeff)
                pcov_w = pcoeff * dstweight_sq

                plap = sp.sparse.csgraph.laplacian(pcov_w.to("cpu").numpy(), normed=True)
                plap = torch.Tensor(plap)

                w, v = torch.lobpcg(plap, k=k, largest=False)
                X = v * w

                # outComp.append(X)
                outComp.append(X)

            outComp = torch.cat(outComp, 0)

            return outComp


        @staticmethod
        def genRepresentationPSI(x_in, k):

            '''cov'''

            #dstweight_sq = UtilsClustering.getDistWeightmap().to(x_in.device)
            dstweight_sq = UtilsClustering.getDistWeightmap()
            vsplit = torch.split(x_in, 19, 0)
            outComp = []

            n = int(x_in.shape[0] / 19)
            for i in range(n):
                # pcoeff = torch.corrcoef(vsplit[i])

                psi = UtilsEdges.computePSI(x_in, "all")
                pcov_w = UtilsTensor.zScoreNorm(psi)
                #pcoeff = UtilsClustering.nonDiagNormalise(pcoeff)
                pcov_w = pcov_w * dstweight_sq

                plap = sp.sparse.csgraph.laplacian(pcov_w.to("cpu").numpy(), normed=True)
                plap = torch.Tensor(plap)

                w, v = torch.lobpcg(plap, k=k, largest=False)
                X = v * w

                # outComp.append(X)
                outComp.append(X)

            outComp = torch.cat(outComp, 0)

            return outComp


        @staticmethod
        def computeLaplacianMSCopilot(adj_matrix):
            # Compute the degree matrix
            D = torch.diag(torch.sum(adj_matrix, dim=1))
            # Compute the Laplacian matrix
            L = D - adj_matrix
            return L

        @staticmethod
        def genRepresentationCau(x_in, k):

            '''cov'''
            #dstweight_sq = UtilsClustering.getDistWeightmap().to(x_in.device)
            dstweight_sq = UtilsClustering.getDistWeightmap()
            vsplit = torch.split(x_in, 19, 0)
            outComp = []

            n = int(x_in.shape[0] / 19)
            for i in range(n):

                df_tigra = DLUtils.genTigraDataFrame(vsplit[i].T, gfp=False)
                pcau = DLUtils.computeCorrPCMI(df_tigra, 2)[:, :, 0]
                pcau = torch.Tensor(pcau)

                #pcov_w = UtilsClustering.nonDiagNormalise(pcov_w)
                pcov_w = pcau * dstweight_sq

                # plap = sp.sparse.csgraph.laplacian(pcov_w.to("cpu").numpy(), normed=True)
                # plap = torch.Tensor(plap)
                plap = UtilsClustering.computeLaplacianMSCopilot(pcov_w)
                w, v = torch.lobpcg(plap, k=k, largest=False)
                X = v * w

                # outComp.append(X)
                outComp.append(X)

            outComp = torch.cat(outComp, 0)

            return outComp

        @staticmethod
        def genRepresentationCauSingle(x_in, k):

            '''cov'''
            #dstweight_sq = UtilsClustering.getDistWeightmap().to(x_in.device)
            dstweight_sq = UtilsClustering.getDistWeightmap()
            #vsplit = torch.split(x_in, 19, 0)
            outComp = []

            n = int(x_in.shape[0] / 19)
            df_tigra = DLUtils.genTigraDataFrame(x_in.T, gfp=False)
            pcau = DLUtils.computeCorrPCMI(df_tigra, 2)[:, :, 0]
            pcau = torch.Tensor(pcau)

            #pcov_w = UtilsClustering.nonDiagNormalise(pcov_w)
            pcov_w = pcau * dstweight_sq

            # plap = sp.sparse.csgraph.laplacian(pcov_w.to("cpu").numpy(), normed=True)
            # plap = torch.Tensor(plap)
            plap = UtilsClustering.computeLaplacianMSCopilot(pcov_w)
            w, v = torch.lobpcg(plap, k=k, largest=False)
            X = v * w

            outComp.append(X)

            outComp = torch.cat(outComp, 0)

            return outComp

        @staticmethod
        def genClusters(X, k):

            vsplit = torch.split(X, 19, 0)
            outComp = []

            n = int(X.shape[0] / 19)

            for i in range(n):
                kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
                kmeans.fit_predict(vsplit[i])
                klabels = kmeans.labels_
                outComp.append(klabels)

            outComp = np.array(outComp)
            return torch.tensor(outComp)

        @staticmethod
        def nonDiagNormalise(pcov):
            pcov = pcov.fill_diagonal_(0, False)
            pcov = pcov/pcov.max()
            pcov = pcov.fill_diagonal_(1, False)

            return pcov

        @staticmethod
        def findLargestClusterIndices(lbls):
            '''Find the largest cluster'''
            n = lbls.shape[0]

            mxclusters = []
            for i in range(n):
                clustercnts = torch.bincount(lbls[i])
                maxcntind = torch.argmax(clustercnts)
                nds_maxcluster = torch.where(lbls[i] == maxcntind)
                mxclusters.append(nds_maxcluster[0])

            return mxclusters

        @staticmethod
        def findSmallestClusterIndices(lbls):
            '''Find the smallest cluster'''
            n = lbls.shape[0]

            mnclusters = []
            for i in range(n):
                clustercnts = torch.bincount(lbls[i])
                mincntind = torch.argmin(clustercnts)
                nds_mincluster = torch.where(lbls[i] == mincntind)
                mnclusters.append(nds_mincluster[0])

            return mnclusters


        @staticmethod
        def filterByIndicesDynamic(x, indices):
            '''Filter x by indices'''
            x_out, h = [], x.shape[0]
            nbs = h // 19
            x = x.view(nbs, 19, -1)

            for i in range(nbs):
                nds = indices[i][torch.where(indices[i]>=0)].tolist()
                #a = x[i][indices[i].tolist()]
                a = x[i][nds]
                x_out.append(a)

           #x_out = torch.stack(x_out)

            return x_out

        @staticmethod
        def filterByIndices(x, indices):
            '''Filter x by indices'''
            x_out, h = [], x.shape[0]
            nbs = h // 19
            x = x.view(nbs, 19, -1)

            for i in range(nbs):
                a = x[i][indices[i].tolist()]
                x_out.append(a)

           #x_out = torch.stack(x_out)

            return x_out



        @staticmethod
        def filterByIndicesDynamic19(x, indices):
            '''Filter x by indices'''
            x_out, h = [], x.shape[0]
            nbs = h // 19
            x = x.view(nbs, 19, -1)

            for i in range(nbs):
                nds = indices[i][torch.where(indices[i]>=0)].tolist()
                #a = x[i][indices[i].tolist()]
                a = x[i][nds]
                x_out.append(a)

           #x_out = torch.stack(x_out)

            return x_out

        @staticmethod
        def filterByIndices19(x, indices):
            '''Filter x by indices'''
            x_out, h = [], x.shape[0]
            nbs = h // 19
            x = x.view(nbs, 19, -1)

            for i in range(nbs):
                a = x[i][indices[i].tolist()]
                x_out.append(a)

           #x_out = torch.stack(x_out)

            return x_out
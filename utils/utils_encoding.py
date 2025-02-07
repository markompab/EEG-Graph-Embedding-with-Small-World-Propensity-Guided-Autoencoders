from collections import Counter

import torch


class UtilsEncoding:

    @staticmethod
    def genClusterCodes(n):
        clustercodes = torch.zeros((n, n))
        for i in range(n):
            clustercodes[i, i] = 1
        return clustercodes

    @staticmethod
    def generateMap(counts, keys, clustercodes):

        count_order = torch.argsort(torch.Tensor(counts) , descending = True)
        clusterDict, i = {}, 0
        for itm in count_order:
            clusterDict[keys[itm]] = clustercodes[i]
            i += 1
        return clusterDict

    @staticmethod
    def generateClusterEncodingMap(lbls):
        lbls = lbls.tolist()
        objCounter = Counter(lbls)
        keys, counts = list(objCounter.keys()), list(objCounter.values())
        n = len(counts)

        clusterCodes = UtilsEncoding.genClusterCodes(n)
        clusterMap = UtilsEncoding.generateMap(counts, keys, clusterCodes)

        return clusterMap

    @staticmethod
    def generateKmeansLblEncodings(lbls):
        lbls = lbls.numpy()
        log = "Kmeans labels: {}".format(lbls)
        print(log)
        '''generate encoding map'''
        map = UtilsEncoding.generateClusterEncodingMap(lbls)

        '''generate encoding'''
        encodings = []

        for lbl in lbls:
            encoding = map[lbl]
            encodings.append(encoding)

        return encodings


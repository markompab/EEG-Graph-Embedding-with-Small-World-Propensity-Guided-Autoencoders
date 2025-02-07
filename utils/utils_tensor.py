import torch


class UtilsTensor:

    @staticmethod
    def zScoreNorm(outmap):
        outmap_min, _ = torch.min(outmap, dim=1, keepdim=True)
        outmap_max, _ = torch.max(outmap, dim=1, keepdim=True)
        outmap = (outmap - outmap_min) / (outmap_max - outmap_min)

        return outmap
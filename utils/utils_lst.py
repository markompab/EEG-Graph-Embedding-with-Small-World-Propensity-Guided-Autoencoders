import math
import torch


class UtilsLst:
    @staticmethod
    def genranges(st, en, numsegs):
        seglen = int((en - st) / numsegs)
        print(seglen)

        rs = []
        for i in range(numsegs):
            r = [st + i * seglen, st + (i + 1) * seglen]
            rs.append(r)

        rs[0][0]  = rs[0][0] - 1
        rs[-1][1] = en + 1

        return rs

    @staticmethod
    def replace_nan_and_inf(torchTensor):
        # Replace NaN values with 0
        torchTensor = torch.nan_to_num(torchTensor, nan=0.0)

        # Find the maximum and minimum finite values (i.e., excluding inf and -inf)
        max_val = torch.max(torchTensor[torch.isfinite(torchTensor)])
        min_val = torch.min(torchTensor[torch.isfinite(torchTensor)])

        # Replace +inf with max finite value
        torchTensor = torch.where(torchTensor == float('inf'), max_val, torchTensor)

        # Replace -inf with min finite value
        torchTensor = torch.where(torchTensor == float('-inf'), min_val, torchTensor)

        return torchTensor
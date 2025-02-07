import torch
import torchaudio


class MetricUtils:
    @staticmethod
    def computeFretchetDistance0(x, y):
        nx = list(range(len(x)))
        x  = torch.squeeze(x)
        x1 = torch.transpose(torch.Tensor([nx , x]), 0, 1)
        mean_x  = torch.mean(x1, 0, False)
        sigma_x = torch.cov(x1)

        ny =  list(range(len(y)))
        y  = torch.squeeze(y)
        y1 = torch.transpose(torch.Tensor([ny , y]), 0, 1)
        mean_y  = torch.mean(y1, 0, False)
        sigma_y = torch.cov(y1)

        dst = torchaudio.functional.frechet_distance(mean_x, sigma_x, mean_y, sigma_y)
        return dst

    @staticmethod
    def computeFretchetDistance(x, y):
        nx = list(range(len(x)))
        x  = torch.squeeze(x)
        x1 = torch.transpose(torch.Tensor([nx , x]), 0, 1)

        ny =  list(range(len(y)))
        y  = torch.squeeze(y)
        y1 = torch.transpose(torch.Tensor([ny , y]), 0, 1)

        return None

    @staticmethod
    def computeAccuracy(x, y, interval=3):
        v = torch.abs(y-x)
        accs = [torch.sum(v<interval)]

        for i in range(1, 4):
            accs.append(torch.sum((v>=i*interval)  & (v<(i+1)*interval)))
        accs.append(torch.sum(v >= interval*4))

        return torch.Tensor(accs)


    @staticmethod
    def computeSimilaritgpt(config):
        best_configuration = [16, 0, 0, 0, 0]
        similarity = 0

        for i in range(len(config)):
            similarity += min(config[i], best_configuration[i])

        return similarity


    @staticmethod
    def computeWeigtedDist5(conf):
        best_conf = torch.Tensor([16, 0, 0, 0, 0])
        w = [1, 0.8, 0.6, 0.4, 0.2]
        n = len(conf)
        dst = 0
        for i in range(n):
            d = torch.abs(conf[i] - best_conf[i])*w[i]
            dst += d

        return dst

    @staticmethod
    def computeWeightedDist(conf):
        best_conf = torch.Tensor([16, 0, 0, 0, 0])
        w = [1, 0.8, 0.6, 0.4, 0.2]
        n = len(conf)
        dst = 0
        for i in range(n):
            d = torch.abs(conf[i] - best_conf[i])*w[i]
            dst += d

        return dst






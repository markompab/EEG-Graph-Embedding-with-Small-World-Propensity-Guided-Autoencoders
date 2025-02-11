import argparse

import math
import os
import torch
#import wandb
import pandas as pd

from torchmetrics import R2Score
from timeit import default_timer as timer
from torch_geometric.loader import DataLoader
from tslearn.metrics import SoftDTWLossPyTorch
from losses.Wingloss import WingLoss
from utils.utils_dl import DLUtils
from utils.utils_plot import PlotUtils
from utils.utils_dttm import UtilsTime
from utils.utils_files import FileUtils
from utils.utils_signal import UtilsSignal
from utils.utils_edges import UtilsEdges
from utils.utils_metrics import MetricUtils
from utils.utils_clustering import UtilsClustering
from models.sagelstmdconv_agh import SageLSTM1DConvAGH
from dataloaders.dl_agh_precomp_swpfreqthresh import DLAGHPSISWPFreqThresh


def torchOptimize():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

torchOptimize()


def loadTrainTestData(srcdir, srcdircorr, srcdircorrswp, srcdir_swpfreqranges, srccsv, csvcol,  nodes, BS=8):

    srcdir_train = srcdir + "train/"
    srcdir_test  = srcdir + "val/"

    dataset_train = DLAGHPSISWPFreqThresh(srcdir_train, srcdir_swpfreqranges, srcdircorr, srcdircorrswp, srccsv,  nodes, column=csvcol, swnlen=SEG)
    dataset_test  = DLAGHPSISWPFreqThresh(srcdir_test , srcdir_swpfreqranges, srcdircorr, srcdircorrswp, srccsv, nodes, column=csvcol, swnlen=SEG)

    # dataset_train = DLRaw(srcdir_train, srccsv, column=csvcol)
    # dataset_test  = DLRaw(srcdir_test, srccsv, column=csvcol)

    train_loader = DataLoader(dataset_train, batch_size=BS, shuffle=True)
    test_loader  = DataLoader(dataset_test, batch_size=BS, shuffle=False)

    return train_loader, test_loader, dataset_train

def splitData(dataset, BS=16):
    ln = len(dataset)
    split = int(ln * (1 - (16 / ln)))

    train_set, test_set = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
    train_loader, test_loader = DataLoader(train_set, batch_size=BS, shuffle=True), DataLoader(test_set, batch_size=BS,
                                                                                               shuffle=False)
    return train_loader, test_loader

def initLogging(modelprops):
    '''Initialize logging'''
    # wandb.init(project="eeg", entity="markompab", config={modelprops: modelprops})

def train(model, optimizer, dataset, train_loader, csvcol, BS, epoch, modelprops, i):
    model.train()
    WL  = WingLoss()
    MAE = torch.nn.L1Loss()
    MSE = torch.nn.MSELoss()
    DTW = SoftDTWLossPyTorch(gamma=1.0)
    preds, gts, serials = [], [], []
    global steps

    train_loss, batch_idx = 0, 0
    x_embs = []

    for data in train_loader:
        graph  = data["graphdata"].to(device)
        gender = data["gender"].to(device)
        handed = data["handed"]
        age    = data["age"]

        if (len(graph.y) != BS):
            continue

        #optimizer.zero_grad()

        '''Fast training'''
        for param in model.parameters():
            param.grad = None

        output, x_emb, x_enc, x_dec = model(graph.x, graph.edge_index, graph.edge_attr, gender, age, handed)
        x_embs.append(x_emb)

        if(len(output) != len(graph.y)):
            continue

        preds.append(output)
        gts.append(graph.y)
        serials.extend(data["serial"])

        wloss = WL(output, graph.y)
        #sm, lg = getSmallWorldNetworks(graph.x)
        #sm, lg = data["swn_sm"], data["swn_lg"]

        # sm, lg = getSmallWorldNetworks(graph.x)

        rw = torch.arange(19)
        sm = rw.repeat(BS, 1)
        freqs = data["swp_freqrange"]

        bp_xorg = UtilsSignal.apply_bandpass_batch(graph.x/torch.max(graph.x), freqs, 500)
        bp_xdec = UtilsSignal.apply_bandpass_batch(x_dec/torch.max(x_dec), freqs, 500)

        x_filtered = UtilsClustering.filterByIndicesDynamic(bp_xorg, sm)
        x_dec_filtered = UtilsClustering.filterByIndicesDynamic(bp_xdec, sm)

        n = len(x_filtered)
        recmse, recdtwdist, l_connectivity = 0, 0, 0
        for i in range(n):
            recmse += MSE(x_filtered[i], x_dec_filtered[i])

            x_in, x_dec = torch.unsqueeze(x_filtered[i], dim=0), torch.unsqueeze(x_dec_filtered[i], dim=0)
            dist = DTW(x_in, x_dec)
            recdtwdist += torch.mean(dist)
            l_connectivity += computeConnectivityLosses(x_filtered[i].T, x_dec_filtered[i].T)

        recmse = recmse / n
        recdtwdist = recdtwdist / n

        w =  torch.softmax(model.loss_weights, dim=0)
        loss = (w[0] * wloss
                + w[1] * recmse
                + w[2] * recdtwdist
                + w[3] * l_connectivity)

        #wandb.log({"recmse_train": recmse, "recdtw_train": recdtwdist, "connectivity_train": l_connectivity, "wloss_train": wloss, "loss_train": loss})
        loss.backward()
        optimizer.step()
        #scheduler.step(loss)

        train_loss += loss
        batch_idx += 1

    gtsc = torch.cat(gts, dim=0).to("cpu")
    gtsc = (gtsc*(dataset.iqmax-dataset.iqmin)) + dataset.iqmin
    gtsc = torch.squeeze(gtsc)

    predsc = torch.cat(preds, dim=0).to("cpu")
    predsc = (predsc*(dataset.iqmax-dataset.iqmin)) + dataset.iqmin
    predsc = torch.squeeze(predsc)

    train_maeloss = MAE(gtsc, predsc).item()
    test_mseloss = MSE(gtsc, predsc).item()
    test_rmse = math.sqrt(test_mseloss)

    # tbwriter.add_scalar('MAE/train', train_maeloss, epoch)
    # tbwriter.add_scalar('Loss/train', train_loss, epoch)

    r2score = R2Score()(gtsc.to('cpu'), predsc.to('cpu')).item()

    if(train_maeloss < 10):
        dstdir = "../runlogs/{}/train/".format(modelprops)

        if(not os.path.exists(dstdir)):
            os.makedirs(dstdir)

        dstpath = "{}/{}_{}".format(dstdir, UtilsTime.cdateTm(), modelprops)
        PlotUtils.plot_eval(predsc.detach(), gtsc.detach(), serials, train_maeloss, csvcol, "{}_train.png".format(dstpath))

        if (train_maeloss < 10):
            '''t-SNE'''
            dstpath_emb = "../runlogs_viz/train/{}_{}.png".format(UtilsTime.cdateTm(),modelprops)
            embs_vstacked = torch.vstack(x_embs)
            embs_vstacked = embs_vstacked.detach().cpu().numpy()
            PlotUtils.visualize_embedding(dstpath_emb, embs_vstacked, n_components=2, perplexity=2, random_state=42)

    log = "{} Epoch {}: Train MAE:{}, RMSE: {} R2 Score:{}".format(i, epoch, train_maeloss, test_rmse,  r2score)
    print(log)

    return model, train_maeloss

@torch.no_grad()
def test(model, dataset, test_loader, BS, epoch, best_test, csvcol, modelprops, i):
    model.eval()
    # Train classifier on training set:
    ''' '''
    target = None
    preds, gts, serials = [], [], []
    MAE = torch.nn.L1Loss()
    MSE = torch.nn.MSELoss()
    WL  = WingLoss()

    DTW = SoftDTWLossPyTorch(gamma=1.0)
    test_loss, batch_idx = 0, 0
    test_wing, batch_idx = 0, 0

    x_embs = []
    with torch.no_grad():
        for data in test_loader:

            graph  = data["graphdata"].to(device)
            gender = data["gender"]
            handed = data["handed"]
            age = data["age"]

            if (len(graph.y) < BS):
                continue

            #output, x_emb, x_enc, x_dec = model(graph.x, graph.edge_index, gender, age, handed)
            output, x_emb, x_enc, x_dec = model(graph.x, graph.edge_index, graph.edge_attr, gender, age, handed)
            x_embs.append(x_emb)

            if (len(output) != len(graph.y)):
                continue

            preds.append(output)
            gts.append(graph.y)
            serials.extend(data["serial"])

            wloss = WL(output, graph.y)
            #sm, lg = getSmallWorldNetworks(graph.x)

            rw = torch.arange(19)
            sm = rw.repeat(BS, 1)
            freqs = data["swp_freqrange"]

            bp_xorg = UtilsSignal.apply_bandpass_batch(graph.x, freqs, 500)
            bp_xdec = UtilsSignal.apply_bandpass_batch(x_dec, freqs, 500)

            x_filtered = UtilsClustering.filterByIndicesDynamic(bp_xorg, sm)
            x_dec_filtered = UtilsClustering.filterByIndicesDynamic(bp_xdec, sm)

            n = len(x_filtered)
            recmse, recdtwdist, l_connectivity = 0, 0, 0
            for i in range(n):
                recmse += MSE(x_filtered[i], x_dec_filtered[i])
                x_in, x_dec = torch.unsqueeze(x_filtered[i], dim=0), torch.unsqueeze(x_dec_filtered[i], dim=0)
                dist = DTW(x_in, x_dec)
                recdtwdist += torch.mean(dist)


            recmse = recmse / n
            recdtwdist = recdtwdist / n
            #w = torch.softmax(model.loss_weights, dim=0)
            # loss = (w[0] * wloss
            #         + w[1] * recmse
            #         + w[2] * recdtwdist
            #         + w[3] * l_connectivity)

            test_loss += wloss
            '''

            wandb.log({
                       "recmse_test": recmse
                     , "recdtw_test": recdtwdist
                    # , "connectivity_train": l_connectivity
                     , "wloss_test": wloss
                     , "loss_test": test_loss
                    }
                )'''

            batch_idx += 1

    gtsc = torch.cat(gts, dim=0).to("cpu")
    gtsc = (gtsc * (dataset.iqmax - dataset.iqmin)) + dataset.iqmin
    gtsc = torch.squeeze(gtsc)

    predsc = torch.cat(preds, dim=0).to("cpu")

    predsc = (predsc * (dataset.iqmax - dataset.iqmin)) + dataset.iqmin
    predsc = torch.squeeze(predsc)

    test_maeloss = MAE(gtsc, predsc).item()
    test_mseloss = MSE(gtsc, predsc).item()
    test_wing = WL(gtsc, predsc)
    test_rmse = math.sqrt(test_mseloss)
    test_acc  = MetricUtils.computeAccuracy(predsc, gtsc)
    test_wdst = MetricUtils.computeWeightedDist(test_acc)

    # tbwriter.add_scalar('MAE/val', test_maeloss, epoch)
    # tbwriter.add_scalar('Loss/val', test_loss, epoch)

    if (test_maeloss < best_test):

        dstdir = "../runlogs/{}/test/".format(modelprops)

        if (not os.path.exists(dstdir)):
            os.makedirs(dstdir)

        dstpath = "{}/{}_{}".format(dstdir, UtilsTime.cdateTm(), modelprops)

        metrics = "{} MAE: {} Wing: {} Acct:{} Acc: {}".format(i,
                                                               round(test_maeloss, 2)
                                                               , round(test_wing.item(), 2)
                                                               , round(test_wdst.item(), 2)
                                                               , test_acc.tolist())

        PlotUtils.plot_eval(predsc.detach(), gtsc.detach(), serials, metrics, csvcol, "{}_val.png".format(dstpath))
        if (test_maeloss < 10):
            '''t-SNE'''
            dstpath_emb = "../runlogs_viz/val/{}_{}.png".format(UtilsTime.cdateTm(),modelprops)
            embs_vstacked = torch.vstack(x_embs)
            embs_vstacked = embs_vstacked.detach().cpu().numpy()

            PlotUtils.visualize_embedding(dstpath_emb, embs_vstacked, n_components=2, perplexity=2, random_state=42)

        # if (test_mael
    if(test_maeloss<10):
        print(test_maeloss)

    log = "Epoch {}: Test MAE:{}, RMSE: {} Acct:{} Acc: {}".format(epoch, round(test_maeloss, 5), round(test_rmse, 5),                                                                   test_wdst, test_acc)
    print(log)

    return test_maeloss, test_loss

def trainvalModel(modelpath, dataset, train_loader, test_loader, modelprops, csvcol, nds, i):

    model = SageLSTM1DConvAGH(1, lenin=8500, BS=BS).to(device)

    if(model is not None):
        model = torch.load(modelpath)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    best_train, best_test, best_acct = 100, 100, 100
    for epoch in range(500):
        model, train_loss   = train(model, optimizer, dataset, train_loader, csvcol, BS, epoch, modelprops, i)
        test_mae, test_wing = test(model, dataset, test_loader, BS, epoch, best_test, csvcol, modelprops, i)
        #scheduler.step(test_mae)

        if(test_mae<best_test):
            best_test = test_mae
            best_train = train_loss
            #torch.save(model, modelpath)

    log = "Best Train MAE: {} Best Test MAE: {}".format(best_train, best_test)
    print(log)
    model = None

    return best_train, best_test

def btchTrainVal(modelpath, srcdir, srcdir_swpfreqranges, srcdircorr, srcdircorrswp, srccsv, titles, ndcmb, modelprops0):

    dstpath_temp = "../model_evalout/{}_{}_temp.csv".format(UtilsTime.cdateTm(), modelprops0)

    torch.manual_seed(0)
    rows = []

    n = len(titles)
    for i in range(n):
        m = len(ndcmb)
        for j in range(m):
            csvcol = titles[i][0]
            nodes = ndcmb[j]
            print(csvcol, nodes)
            train_loader, test_loader, dataset = loadTrainTestData(srcdir, srcdircorr, srcdircorrswp, srcdir_swpfreqranges, srccsv, csvcol, nodes, BS=BS)

            strnds = (str(nodes).replace("'", "")
                                   .replace(",","_")
                                   .replace("[","")
                                   .replace("]","")
                                   .replace(" ",""))
            if(len(nodes) >= 19):
                strnds = "all"

            modelprops = "{}_{}_{}_nds_{}".format(modelprops0, dataset.getLabel(), csvcol, strnds)
            best_train, best_test = trainvalModel(modelpath, dataset, train_loader, test_loader, modelprops, csvcol, nodes, i)
            row = [
                csvcol,
                str(nodes),
                best_train,
                best_test
            ]
            rows.append(row)

            out_row = ",".join([str(a) for a in row])
            with open(dstpath_temp, "a") as f:
                f.write(out_row + "\n")

    header = ["CSV Column", "Nodes", "Best Train", "Best Test"]
    df = pd.DataFrame(rows, columns=header)

    dstpath = "../model_evalout/{}_{}.csv".format(UtilsTime.cdateTm(), modelprops0)
    df.to_csv(dstpath, index=False)

def computeConnectivityLosses(x,y):
    ''''''
    taumax = 10
    n = x.shape[1]
    x, y = x.T.cpu().detach().numpy(), y.T.cpu().detach().numpy()
    x_edges, x_attr = DLUtils.genEdgesPSI(x)
    x_cau = UtilsEdges.edge2Adj(x_edges, x_attr, n)

    y_edges, y_attr = DLUtils.genEdgesPSI(y)
    y_cau = UtilsEdges.edge2Adj(y_edges, y_attr, n)

    tx_cau, ty_cau = torch.tensor(x_cau), torch.tensor(y_cau)
    loss_frobenius = torch.norm(tx_cau - ty_cau, p='fro')

    COSSIM = torch.nn.CosineSimilarity()
    loss_cosine = 2 - (torch.mean(COSSIM(tx_cau, ty_cau))+1)

    loss = loss_frobenius + loss_cosine

    return loss

titles = [
    ["birth_months", 68]
    # , ["reception_months", 76]
    # , ["reception_score", 50]
    # , ["reception_score_months", 68]
    # , ["expression_months", 76]
    # , ["expression_months", 76]
    # , ["expression_score", 54]
    # , ["expression_score_months", 72]
    # , ["total_months", 73]
    # , ["verbal_comprehension", 136]
     # ["iq", 138]
]


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#model2 = GCNRawFull(1, lenin=8500, k=20).to(device)
#model2 = GCNSageLSTMRawPluginAgeGenderHanded(1, lenin=8500).to(device)
#model2 = GCNSageLSTMRawPluginAGHOneHotWRec(1, lenin=8500).to(device)
stdatetm = UtilsTime.cdateTm()

start = timer()

# Argument parsing
parser = argparse.ArgumentParser(description="EEG Model Training Script")
parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
parser.add_argument('--seg', type=int, required=True, help='Segment')
parser.add_argument('--root_dir', type=str, required=True, help='Root directory for EEG data')
parser.add_argument('--root_log', type=str, required=True, help='Root log name')
parser.add_argument('--srccsv', type=str, required=True, help='Source CSV file path')
parser.add_argument('--srcdircorr', type=str, required=True, help='Source directory for corrected data')
parser.add_argument('--srcdir_swpfreqranges', type=str, required=True, help='Source directory for frequency ranges')
parser.add_argument('--srcdircorrswp', type=str, required=True, help='Source directory for precomputed SWP data')
parser.add_argument('--modelpath', type=str, required=True, help='Path to the model file')

args = parser.parse_args()

# Assigning parsed arguments to variables
BS = args.batch_size
SEG = args.seg
root_dir = args.root_dir
root_log = args.root_log
srccsv = args.srccsv
srcdircorr = args.srcdircorr
srcdir_swpfreqranges = args.srcdir_swpfreqranges
srcdircorrswp = args.srcdircorrswp
modelpath = args.modelpath

for i in range(1,5):
    srcdir = "/{}/{}/".format(root_dir, i - 1)
    modelprops0 = "{}_fx{}_plugin_aghatt_{}_lossrecons_swpfreqthresh_reconly_swpthreshfreqnds_connpsi_norm_emb".format(root_log, i ,BS)
    #initLogging(modelprops0)

    ndcmbs_f1 = FileUtils.parseNodeFile("../dataconf/5fold_sp/nds_f1_all.txt".format(i))
    btchTrainVal(modelpath, srcdir, srcdir_swpfreqranges, srcdircorr, srcdircorrswp, srccsv, titles, ndcmbs_f1, modelprops0)

    print("_________________________________________________________________________")

    #wandb.finish()

end = timer()
log ="Run time: {}".format(start-end)
print(log)
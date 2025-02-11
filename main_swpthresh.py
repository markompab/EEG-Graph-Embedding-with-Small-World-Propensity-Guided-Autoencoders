import math
import os

import torch
import wandb
import numpy as np
import pandas as pd
import networkx as nx

from torchmetrics import R2Score
from timeit import default_timer as timer
from torch_geometric.loader import DataLoader
from tslearn.metrics import SoftDTWLossPyTorch
from torch.utils.tensorboard import SummaryWriter

from losses.Wingloss import WingLoss
from utils.utils_dl import DLUtils
from utils.utils_plot import PlotUtils
from utils.utils_dttm import UtilsTime
from utils.utils_files import FileUtils
from utils.utils_metrics import MetricUtils
from utils.utils_clustering import UtilsClustering
from models.sagelstmdconv_agh import SageLSTM1DConvAGH
from dataloaders.dl_eeg_raw_subgraph_plugiagh_onehot_xt_n60_org_precomp_recswpthresh_fst import DLSubGPlgGAHNorm60OrgPSIAllRecSWPThreshPreFst


def torchOptimize():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

torchOptimize()

def dtw(y_true, y_pred):
    dtw_distance, _ = fastdtw(y_true.cpu().numpy(), y_pred.cpu().numpy())
    return dtw_distance

def edge2Adj(edges, edgeweights, n):

    adj = np.zeros((n, n))
    for i in range(edges.shape[1]):
        adj[edges[0, i], edges[1, i]] = edgeweights[i]
        adj[edges[1, i], edges[0, i]] = edgeweights[i]

    return adj

def computeConnectivityLosses(x,y):
    ''''''
    taumax = 10
    n = x.shape[1]
    x, y = x.T.cpu().detach().numpy(), y.T.cpu().detach().numpy()
    x_edges, x_attr = DLUtils.genEdgesPSI(x)
    x_cau =  edge2Adj(x_edges, x_attr, n)

    y_edges, y_attr = DLUtils.genEdgesPSI(y)
    y_cau = edge2Adj(y_edges, y_attr, n)

    tx_cau, ty_cau = torch.tensor(x_cau), torch.tensor(y_cau)
    loss_frobenius = torch.norm(tx_cau - ty_cau, p='fro')

    COSSIM = torch.nn.CosineSimilarity()
    loss_cosine = 2 - (torch.mean(COSSIM(tx_cau, ty_cau))+1)

    loss = loss_frobenius + loss_cosine

    return loss

def genGrid(grid_size=100):
    pos = [
         [-0.30699, 0.94483]  # Fp2
        ,  [0.30699, 0.94483]  # Fp1

        , [-0.80377, 0.58397]  # F8
        , [-0.56611, 0.69909]  # F4
        , [0, 0.76041]  # Fz
        , [0.56611, 0.69909]  # F3
        , [0.80377, 0.58397]  # F7

        , [-0.99361, 0]       #T8
        , [-0.75927, 0.00133] #C4
        , [0, 0.00175]        #Cz
        , [0.75927, 0.00133]  #C3
        , [0.99361, 0]        #T7

        , [-0.80393, -0.58409]#P8
        , [-0.56563, -0.69849]#P4
        , [0,        -0.75813]#Pz
        , [0.56563,  -0.69849]#P3
        , [0.80393,  -0.58409]#P7

        , [-0.30709, -0.94513]#O2
        , [0.30709,  -0.94513]#O1
    ]
    pos = np.array(pos)
    #pos =  np.flip(pos, axis=None)
    #pos[:,1] =  1-pos[:,1]
    #pos[1, :] = 1 - pos[1, :]
    return pos

def genNetxGraph(nodes, edges, edge_weight, pos):

    G = nx.Graph()
    sh = edges.shape

    for i in range(len(nodes)):
        G.add_node(nodes[i], pos=pos[i])

    for i in range(sh[0]):
        #if(edges[i][0] == edges[i][1]):
        #    continue

        if(edge_weight[i] < 0.3):
            continue

        G.add_edge(edges[i][0], edges[i][1], weight=edge_weight[i])

    return G

def computedAverageNdShtstPathLength(node, path_lengths):
  total_path_length = 0
  neighbor_count = len(path_lengths[node]) - 1  # Exclude itself
  for neighbor, path_length in path_lengths[node].items():
    total_path_length += path_length
  return total_path_length / neighbor_count if neighbor_count > 0 else 0  # Handle isolated nodes

def computeAverageNdShtstPathLengthAllNds(G):
    # Calculate all pairs shortest path lengths
    all_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    avgpPathLengths = []
    for node in G.nodes():
        avg_path_length = computedAverageNdShtstPathLength(node, all_path_lengths)
        avgpPathLengths.append(avg_path_length)

    return avgpPathLengths

def initLogging(modelprops):
    wandb.init(project="eeg", entity="markompab",
               config={modelprops: modelprops})

def getNodeLabels(sm):
    n = len(sm)
    NDS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
    smNds = []
    for i in range(n):
        m = len(sm[i])
        for j in range(m):
            smnd =  NDS[sm[i][j]]
            smNds.append(smnd)

    return smNds

def train(model, optimizer, dataset, train_loader, csvcol, BS, epoch, modelprops, i):
    model.train()
    WL  = WingLoss()
    MAE = torch.nn.L1Loss()
    MSE = torch.nn.MSELoss()
    DTW = SoftDTWLossPyTorch(gamma=1.0)
    preds, gts, serials = [], [], []
    global steps

    train_loss, batch_idx = 0, 0

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

        if(len(output) != len(graph.y)):
            continue

        preds.append(output)
        gts.append(graph.y)
        serials.extend(data["serial"])

        wloss = WL(output, graph.y)
        #sm, lg = getSmallWorldNetworks(graph.x)
        #sm, lg = data["swn_sm"], data["swn_lg"]
        sm = data["swn_graph"]

        #log = getNodeLabels(sm)
        #print("Train Nodes:{} ","|".join(log))
        #tm_filt_st = datetime.now()

        x_filtered = UtilsClustering.filterByIndicesDynamic(graph.x, sm)
        x_dec_filtered = UtilsClustering.filterByIndicesDynamic(x_dec, sm)

        n = len(x_filtered)
        recmse, recdtwdist, l_connectivity = 0, 0, 0
        for i in range(n):
            recmse += MSE(x_filtered[i], x_dec_filtered[i])
            #x_in, x_dec = toBatches(x_filtered[i], 19), toBatches(x_dec_filtered[i], 19)

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

        wandb.log({"recmse_train": recmse, "recdtw_train": recdtwdist, "connectivity_train": l_connectivity, "wloss_train": wloss, "loss_train": loss})
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
    with torch.no_grad():
        total_loss = total_examples = 0
        #for data in tqdm.tqdm(test_dataset):
        for data in test_loader:

            graph  = data["graphdata"].to(device)
            gender = data["gender"]
            handed = data["handed"]
            age = data["age"]

            if (len(graph.y) < BS):
                continue

            #output, x_emb, x_enc, x_dec = model(graph.x, graph.edge_index, gender, age, handed)
            output, x_emb, x_enc, x_dec = model(graph.x, graph.edge_index, graph.edge_attr, gender, age, handed)

            if (len(output) != len(graph.y)):
                continue

            preds.append(output)
            gts.append(graph.y)
            serials.extend(data["serial"])

            wloss = WL(output, graph.y)
            #sm, lg = getSmallWorldNetworks(graph.x)
            sm = data["swn_graph"]

            # log = getNodeLabels(sm)
            # print("Test Nodes:{}", log)

            x_filtered = UtilsClustering.filterByIndicesDynamic(graph.x, sm)
            x_dec_filtered = UtilsClustering.filterByIndicesDynamic(x_dec, sm)

            n = len(x_filtered)
            recmse, recdtwdist, l_connectivity = 0, 0, 0
            for i in range(n):
                recmse += MSE(x_filtered[i], x_dec_filtered[i])
                #x_in, x_dec = toBatches(x_filtered[i], 19), toBatches(x_dec_filtered[i], 19)
                x_in, x_dec = torch.unsqueeze(x_filtered[i], dim=0), torch.unsqueeze(x_dec_filtered[i], dim=0)
                dist = DTW(x_in, x_dec)
                recdtwdist += torch.mean(dist)

                #l_connectivity += computeConnectivityLosses(x_filtered[i].T, x_dec_filtered[i].T)


            recmse = recmse / n
            recdtwdist = recdtwdist / n
            #w = torch.softmax(model.loss_weights, dim=0)
            # loss = (w[0] * wloss
            #         + w[1] * recmse
            #         + w[2] * recdtwdist
            #         + w[3] * l_connectivity)

            test_loss += wloss

            wandb.log({
                       "recmse_test": recmse
                     , "recdtw_test": recdtwdist
                    # , "connectivity_train": l_connectivity
                     , "wloss_test": wloss
                     , "loss_test": test_loss
                    }
                )

            batch_idx += 1

    print("preds: {}".format(len(preds)))
    print("gts: {}".format(len(gts)))
    print(modelprops)

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

        # if (test_mael
    if(test_maeloss<10):
        print(test_maeloss)

    log = "Epoch {}: Test MAE:{}, RMSE: {} Acct:{} Acc: {}".format(epoch, round(test_maeloss, 5), round(test_rmse, 5),                                                                   test_wdst, test_acc)
    print(log)

    return test_maeloss, test_loss
    #return test_maeloss, test_wing

def trainvalModel(dataset, train_loader, test_loader, modelprops, csvcol, nds, i):
    #modelpath = "../models_pt/{}_{}.pt".format(UtilsTime.cdateTm(), modelprops)
    modelpath = "../models_pt/{}_{}.pt".format(stdatetm, modelprops)

    model = SageLSTM1DConvAGH(1, lenin=8500, BS=BS).to(device)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

def splitData(dataset, BS=16):
    ln = len(dataset)
    split = int(ln * (1 - (16 / ln)))

    train_set, test_set = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
    train_loader, test_loader = DataLoader(train_set, batch_size=BS, shuffle=True), DataLoader(test_set, batch_size=BS,
                                                                                               shuffle=False)
    return train_loader, test_loader

def loadTrainTestData(srcdir, srcdircorr, srcdirswp,  srccsv, csvcol,  nodes, BS=8):

    srcdir_train = srcdir + "train/"
    srcdir_test  = srcdir + "val/"

    dataset_train = DLSubGPlgGAHNorm60OrgPSIAllRecSWPThreshPreFst(srcdir_train, srcdircorr, srcdirswp, srccsv,  nodes, column=csvcol, swnlen=SEG)
    dataset_test  = DLSubGPlgGAHNorm60OrgPSIAllRecSWPThreshPreFst(srcdir_test, srcdircorr, srcdirswp, srccsv, nodes, column=csvcol, swnlen=SEG)
    # dataset_train = DLRaw(srcdir_train, srccsv, column=csvcol)
    # dataset_test  = DLRaw(srcdir_test, srccsv, column=csvcol)

    train_loader = DataLoader(dataset_train, batch_size=BS, shuffle=True)
    test_loader  = DataLoader(dataset_test, batch_size=BS, shuffle=False)

    return train_loader, test_loader, dataset_train

def btchTrainVal(srcdir, srcdircorr, srcdircorrswp, srccsv, titles, ndcmb, modelprops0):

    #dstpath = "../model_evalout/{}_{}.csv".format(UtilsTime.cdateTm(), modelprops0)
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
            train_loader, test_loader, dataset = loadTrainTestData(srcdir, srcdircorr, srcdircorrswp, srccsv, csvcol, nodes, BS=BS)

            strnds = (str(nodes).replace("'", "")
                                   .replace(",","_")
                                   .replace("[","")
                                   .replace("]","")
                                   .replace(" ",""))
            if(len(nodes) >= 19):
                strnds = "all"

            modelprops = "{}_{}_{}_nds_{}".format(modelprops0, dataset.getLabel(), csvcol, strnds)
            best_train, best_test = trainvalModel(dataset, train_loader, test_loader, modelprops, csvcol, nodes, i)
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

def filterByKLabels(x, klabels, k):
    '''identify largest cluster in klabels'''
    '''Filter x by klabels == k'''
    klabels = klabels.to("cpu")
    klabels = torch.squeeze(klabels)
    klabels = klabels == k
    klabels = klabels.to("cuda")
    return x[klabels]

def spectralRecLoss(x, y, nds):
    '''clustering'''

def toBatches(x, ndCnt):
    n = int(x.shape[0] / ndCnt)
    batches = []
    for i in range(n):
        batch = x[i * ndCnt:(i + 1) * ndCnt, :]
        batches.append(batch)

    return torch.stack(batches)

titles = [
    ["birth_months", 68]
    , ["reception_months", 76]
    , ["reception_score", 50]
    , ["reception_score_months", 68]
    , ["expression_months", 76]
    , ["expression_score", 54]
    , ["expression_score_months", 72]
    , ["total_months", 73]
    #, ["verbal_comprehension", 136]
     # ["iq", 138]
]
#BS=8
BS=16
SEG = 1
#BS=2
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#model2 = GCNRawFull(1, lenin=8500, k=20).to(device)
#model2 = GCNSageLSTMRawPluginAgeGenderHanded(1, lenin=8500).to(device)
#model2 = GCNSageLSTMRawPluginAGHOneHotWRec(1, lenin=8500).to(device)
stdatetm = UtilsTime.cdateTm()

start = timer()

root_dir = "/home/cvnar/09.EEG/04.Datasets/eeg_raw/eeg_data_5fold_sp/"
root_log = "_SAGE_LSTM_SubGraphs_5f_all_dcorrected_5fold_sp"

srccsv = "/home/cvnar/09.EEG/04.Datasets/meta/iq_info_corrected.csv"

srcdircorr = "/home/cvnar/09.EEG/04.Datasets/corr_precomp/pcmci/"
srcdircorrswp = "/home/cvnar/09.EEG/04.Datasets/swp_precomp_25to75_2/pcmci/"
#srcmeta = "/media/cvnar/e2205367-276b-4580-8a1f-79389838ff35/04.EEG//meta/rawfile_meta2.csv"

# srccsv = "/home/cvnar/09.EEG/04.Datasets/meta/iq_info_extended.csv"

for i in range(1,6):
    srcdir = "/{}/{}/".format(root_dir, i - 1)
    modelprops0 = "{}_fx{}_BS{}_plugin_aghatt_recon_only_prepcmci_25to75_connpsi_o".format(root_log, i ,BS)
    initLogging(modelprops0)

    ndcmbs_f1 = FileUtils.parseNodeFile("../dataconf/5fold_sp/nds_f1_all.txt".format(i))
    btchTrainVal(srcdir, srcdircorr, srcdircorrswp, srccsv, titles, ndcmbs_f1, modelprops0)
    print("_________________________________________________________________________")

    wandb.finish()

end = timer()
log ="Run time: {}".format(start-end)
print(log)
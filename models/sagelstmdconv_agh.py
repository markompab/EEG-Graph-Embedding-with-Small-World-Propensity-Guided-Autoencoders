import torch
import torch.nn.functional as F
from torch.nn import LSTM
from torch_geometric.nn import GCNConv, Linear, MLP, global_add_pool, SAGEConv


class SageLSTM1DConvAGH(torch.nn.Module):
    def __init__(self, out_channels, lenin=8500,  BS=16):
        super().__init__()

        self.BS = BS

        #self.conv1 = GCNConv(in_channels=1280, out_channels=640)
        self.lstm = LSTM(input_size=lenin, hidden_size=640, num_layers=3)
        #self.gcn1 = SAGEConv(in_channels=640, out_channels=320)
        self.gcn1 = SAGEConv(in_channels=640, out_channels=480)
        self.conv1 = torch.nn.Conv1d(in_channels=BS*19, out_channels=BS*19, kernel_size=161, stride=1)

        #self.gcn2 = SAGEConv(in_channels=320, out_channels=180)
        self.gcn2 = SAGEConv(in_channels=320, out_channels=250)
        self.conv2 = torch.nn.Conv1d(in_channels=BS*19, out_channels=BS*19, kernel_size=71, stride=1)

        #self.gcn3 = SAGEConv(in_channels=180, out_channels=90)
        self.gcn3 = SAGEConv(in_channels=180, out_channels=135)
        self.conv3 = torch.nn.Conv1d(in_channels=BS*19, out_channels=BS*19, kernel_size=46, stride=1)

        #self.gcn4 = SAGEConv(in_channels=90, out_channels=50)
        self.gcn4 = SAGEConv(in_channels=90, out_channels=70)
        self.conv4 = torch.nn.Conv1d(in_channels=BS*19, out_channels=BS*19, kernel_size=21, stride=1)
        #
        #
        # self.decgcn1 = GCNConv(in_channels=50, out_channels=90)
        # self.decgcn2 = GCNConv(in_channels=90, out_channels=180)
        # self.decgcn3 = GCNConv(in_channels=180, out_channels=320)
        # self.decgcn4 = GCNConv(in_channels=320, out_channels=640)
        #
        self.decgcn2 = SAGEConv(in_channels=50, out_channels=180)
        self.decgcn4 = SAGEConv(in_channels=180, out_channels=640)

        # self.declstm = LSTM(input_size=640, hidden_size=lenin, num_layers=3)
        self.declstm = LSTM(input_size=640, hidden_size=lenin, num_layers=1)

        self.fcn1  = Linear(in_channels=58, out_channels=32)
        self.fcn2  = Linear(in_channels=32, out_channels=16)
        self.fcn3  = Linear(in_channels=16, out_channels=1)
        #self.MLP = MLP([1024, 512, 256, out_channels], dropout=0.5, norm=None)

        weights_init = torch.Tensor([1.0, 1.0, 0.01, 0.1]).requires_grad_(True)
        #self.loss_weights = torch.nn.Parameter(torch.ones(4, requires_grad=True))
        self.loss_weights = torch.nn.Parameter(weights_init)


    def get_batch(self, x, BS=16):
        btch, n = [], x.shape[0]

        blen =  int(n / BS)

        for i in range(n):
            lbl = int(i / blen)
            btch.append(lbl)

        return torch.Tensor(btch).type(torch.int64)

    def forward(self, x_in, edge_index, edge_weight, gender, age, handed):
        edge_index = edge_index.type(torch.int64)
        #unning_mean = torch.mean(x, dim=1)
        #running_var = torch.var(x, dim= 1)

        '''encoder'''
        x = self.lstm(x_in)[0]

        if torch.isnan(x).any():
            print("Nan found")

        x_enc = self.forwardEncoderGCN(x, edge_index, edge_weight)

        if torch.isnan(x).any():
            print("Nan found")

        '''decoder'''
        x_dec = self.forwardDecoderGCN(x_enc, edge_index, edge_weight)
        x_dec = self.declstm(x_dec)[0]

        if torch.isnan(x_dec).any():
            print("Nan found")

        batch = self.get_batch(x_enc, BS=self.BS).to(x.device)
        gender = gender.to(x.device)
        handed = handed.to(x.device)
        age = age.to(x.device)

        x = global_add_pool(x_enc, batch)
        x = torch.concatenate([x, gender, age, handed], dim=1)

        x_emb = x.clone()

        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        #x = torch.sigmoid(x)
        #return torch.log_softmax(x, dim=1)
        return x, x_emb, x_enc, x_dec

    def forwardEncoderGCN(self, x_in, edge_index, edge_weight):
        #x_enc = self.gcn1(x_in, edge_index, edge_weight)
        print("x_in", x_in.shape)
        print("edge_index", edge_index.shape)
        x_enc = self.gcn1(x_in, edge_index)
        x_enc = self.conv1(x_enc)
        x_enc = F.leaky_relu(x_enc)
        x_enc = F.batch_norm(x_enc, None, None, training=True)

        #x_enc = self.gcn2(x_enc, edge_index, edge_weight)
        x_enc = self.gcn2(x_enc, edge_index)
        x_enc = self.conv2(x_enc)
        x_enc = F.leaky_relu(x_enc)
        x_enc = F.batch_norm(x_enc, None, None, training=True)

        #x_enc = self.gcn3(x_enc, edge_index, edge_weight)
        x_enc = self.gcn3(x_enc, edge_index)
        x_enc = self.conv3(x_enc)
        x_enc = F.leaky_relu(x_enc)
        x_enc = F.batch_norm(x_enc, None, None, training=True)

        #x_enc = self.gcn4(x_enc, edge_index, edge_weight)
        x_enc = self.gcn4(x_enc, edge_index)
        x_enc = self.conv4(x_enc)
        x_enc = F.leaky_relu(x_enc)
        x_enc = F.batch_norm(x_enc, None, None, training=True)

        return x_enc

    def forwardDecoderGCN(self, x_in, edge_index, edge_weight):
        # x_dec = self.decgcn1(x_in, edge_index, edge_weight)
        # x_dec = F.leaky_relu(x_dec)
        # x_dec = F.batch_norm(x_dec, None, None, training=True)

        #x_dec = self.decgcn2(x_in, edge_index, edge_weight)
        x_dec = self.decgcn2(x_in, edge_index)
        x_dec = F.leaky_relu(x_dec)
        x_dec = F.batch_norm(x_dec, None, None, training=True)

        # x_dec = self.decgcn3(x_dec, edge_index, edge_weight)
        # x_dec = F.leaky_relu(x_dec)
        # x_dec = F.batch_norm(x_dec, None, None, training=True)
        #
        #x_dec = self.decgcn4(x_dec, edge_index, edge_weight)
        x_dec = self.decgcn4(x_dec, edge_index)
        x_dec = F.leaky_relu(x_dec)
        x_dec = F.batch_norm(x_dec, None, None, training=True)

        return x_dec

    def getLabel(self):
        lbl = "SageConvLSTMRawPluginAGHOneHotWRecGLCNNTParams"
        return lbl

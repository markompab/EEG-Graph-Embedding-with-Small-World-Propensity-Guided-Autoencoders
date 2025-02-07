import torch
import torch.nn.functional as F
from torch.nn import LSTM
from torch_geometric.nn import GCNConv, Linear, MLP, global_add_pool, SAGEConv


class SAGEGCNLSTMRawPluginAgeGender(torch.nn.Module):
    def __init__(self, out_channels, lenin=8500,  BS=16):
        super().__init__()

        self.BS = BS

        #self.conv1 = GCNConv(in_channels=1280, out_channels=640)
        self.lstm = LSTM(input_size=lenin, hidden_size=640, num_layers=3)
        self.gcn1 = SAGEConv(in_channels=640, out_channels=320)
        self.gcn2 = SAGEConv(in_channels=320, out_channels=180)
        self.gcn3 = SAGEConv(in_channels=180, out_channels=90)
        self.gcn4 = SAGEConv(in_channels=90, out_channels=50)
        self.fcn1  = Linear(in_channels=54, out_channels=32)
        self.fcn2  = Linear(in_channels=32, out_channels=16)
        self.fcn3  = Linear(in_channels=16, out_channels=1)
        self.MLP = MLP([1024, 512, 256, out_channels], dropout=0.5, norm=None)

    def get_batch(self, x, BS=16):
        btch, n = [], x.shape[0]

        blen =  int(n / BS)

        for i in range(n):
            lbl = int(i / blen)
            btch.append(lbl)

        return torch.Tensor(btch).type(torch.int64)

    def forward(self, x_in, edge_index, gender, age):

        #unning_mean = torch.mean(x, dim=1)
        #running_var = torch.var(x, dim= 1)

        x = self.lstm(x_in)[0]

        x = self.gcn1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.batch_norm(x, None, None,  training=True)

        x = self.gcn2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.batch_norm(x, None, None, training=True)

        x = self.gcn3(x, edge_index)
        x = F.leaky_relu(x)
        x = F.batch_norm(x, None, None, training=True)

        x = self.gcn4(x, edge_index)
        x = F.leaky_relu(x)
        x = F.batch_norm(x, None, None, training=True)

        batch = self.get_batch(x, BS=self.BS).to(x.device)
        gender = gender.to(x.device)
        age = age.to(x.device)

        x = global_add_pool(x, batch)
        x = torch.concatenate([x, gender, age], dim=1)

        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        #x = torch.sigmoid(x)
        #return torch.log_softmax(x, dim=1)
        return x


    def getLabel(self):
        lbl = "GCNLSTMRawPluginAgeGender"
        return lbl
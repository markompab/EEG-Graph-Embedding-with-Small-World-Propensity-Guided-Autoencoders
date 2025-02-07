import torch
import torch.nn.functional as F
from torch.nn import LSTM
from torch_geometric.nn import GCNConv, Linear, MLP, global_add_pool, SAGEConv


class GCNSageLSTMRawPluginAgeGenderHandedOneHotAtt(torch.nn.Module):
    def __init__(self, out_channels, lenin=8500,  BS=16):
        super().__init__()

        self.BS = BS

        #self.conv1 = GCNConv(in_channels=1280, out_channels=640)
        self.lstm = LSTM(input_size=lenin, hidden_size=640, num_layers=3)
        self.gcn1 = SAGEConv(in_channels=640, out_channels=320)
        self.gcn2 = SAGEConv(in_channels=320, out_channels=180)
        self.gcn3 = SAGEConv(in_channels=180, out_channels=90)
        self.gcn4 = SAGEConv(in_channels=90, out_channels=50)
        self.fcn1 = Linear(in_channels=54, out_channels=32)
        self.fcn2 = Linear(in_channels=32, out_channels=16)
        self.fcn3 = Linear(in_channels=16, out_channels=1)
        self.MLP = MLP([1024, 512, 256, out_channels], dropout=0.5, norm=None)
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=50, num_heads=5, dropout=0.5)
        self.initAttention()

    def initAttention(self):

        #d_model, heads = 1280, 1
        d_model, heads = 5, 4
        self.d_model = d_model  # embedding dimension
        self.heads = heads  # number of attention heads

        self.query = torch.nn.Linear(d_model, d_model * heads)
        self.key   = torch.nn.Linear(d_model, d_model * heads)
        self.value = torch.nn.Linear(d_model, d_model * heads)
        self.fc    = torch.nn.Linear(d_model * heads, 4)
        self.relu3 = torch.nn.ReLU()


    def get_batch(self, x, BS=16):
        btch, n = [], x.shape[0]

        blen =  int(n / BS)

        for i in range(n):
            lbl = int(i / blen)
            btch.append(lbl)

        return torch.Tensor(btch).type(torch.int64)

    def forward(self, x_in, edge_index, gender, age, handed):
        edge_index = edge_index.type(torch.int64)
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
        handed = handed.to(x.device)
        age = age.to(x.device)

        x = global_add_pool(x, batch)
        plug=  torch.concatenate([gender, age, handed], dim=1)
        plugx = self.forward_attention(plug)
        x = torch.concatenate([x, plugx], dim=1)
        x_emb = x.clone()

        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        #x = torch.sigmoid(x)
        #return torch.log_softmax(x, dim=1)
        return x, x_emb


    def  forward_attention(self, x_in):
        #n = (int(x_in.shape[0]/19))
        #x = torch.reshape(x_in,[n, 19, 19])
        x = x_in
        # Project to query, key and value
        q = self.query(x)  # (batch_size, seq_len, d_model * heads)
        k = self.key(x)  # (batch_size, seq_len, d_model * heads)
        v = self.value(x)  # (batch_size, seq_len, d_model * heads)

        # Split into heads
        q_heads = q.view(x.size(0), -1, self.heads, self.d_model)  # (batch_size, seq_len, heads, d_head)
        k_heads = k.view(x.size(0), -1, self.heads, self.d_model)  # (batch_size, seq_len, heads, d_head)
        v_heads = v.view(x.size(0), -1, self.heads, self.d_model)  # (batch_size, seq_len, heads, d_head)

        d_modelt = torch.tensor([self.d_model]).to(x.device)

        # Attention score
        scores = torch.matmul(q_heads, k_heads.transpose(2, 3)) / torch.sqrt(d_modelt)  # (batch_size, seq_len, heads, seq_len)

        # Softmax and scale
        #attention = torch.nn.Softmax(dim=-1)(scores) * torch.sqrt(d_modelt)  # (batch_size, seq_len, heads, seq_len)
        attention = torch.nn.functional.gumbel_softmax(scores.squeeze(), tau=1 ,  hard=False)  * torch.sqrt(d_modelt)# (batch_size, seq_len, heads, seq_len)
        attention = attention.view(self.BS, 1, self.heads, -1)

        # Weighted sum of values
        output = torch.matmul(attention, v_heads)  # (batch_size, seq_len, heads, d_head)

        # Concatenate heads and project
        output = output.view(x.size(0), x.size(1), -1)  # (batch_size, seq_len, d_model)
        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)  # (batch_size, seq_len, d_model)
        #output = torch.nn.functional.gumbel_softmax(output.squeeze(), tau=1 ,  hard=False)

        return output


    def getLabel(self):
        lbl = "GCNSageLSTMRawPluginAgeGenderHandedOneHotAtt"
        return lbl
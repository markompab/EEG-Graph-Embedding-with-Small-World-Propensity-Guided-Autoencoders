import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from reformer_pytorch import ReformerLM, Autopadder, Reformer
from torch.nn import LSTM
from torch_geometric.nn import GCNConv, Linear, MLP, global_add_pool, SAGEConv


class SageReformerLSTMRawPluginAgeGenderHandedOneHotAttFast(torch.nn.Module):
    def __init__(self, out_channels, lenin=8192,  BS=16):
        super().__init__()

        self.BS = BS
        self.reformer_out_channels = 640

        #self.conv1 = GCNConv(in_channels=1280, out_channels=640)
        self.lstm = LSTM(input_size=lenin, hidden_size=640, num_layers=3)
        self.gcn1 = GCNConv(in_channels=self.reformer_out_channels, out_channels=320)
        self.gcn2 = SAGEConv(in_channels=320, out_channels=180)
        self.gcn3 = SAGEConv(in_channels=180, out_channels=90)
        self.gcn4 = SAGEConv(in_channels=90, out_channels=50)
        self.fcn1 = Linear(in_channels=54, out_channels=32)
        self.fcn2 = Linear(in_channels=32, out_channels=16)
        self.fcn3 = Linear(in_channels=16, out_channels=1)
        self.MLP = MLP([1024, 512, 256, out_channels], dropout=0.5, norm=None)
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=50, num_heads=5, dropout=0.5)
        self.initAttention()

        self.reformer_ap = self.initReformer()
        self.maxlen = 8192

    def initAttention(self):

        #d_model, heads = 1280, 1
        d_model, heads = 8, 4
        self.d_model = d_model  # embedding dimension
        self.heads = heads  # number of attention heads

        self.query = torch.nn.Linear(d_model, d_model * heads)
        self.key   = torch.nn.Linear(d_model, d_model * heads)
        self.value = torch.nn.Linear(d_model, d_model * heads)
        self.fc    = torch.nn.Linear(d_model * heads, 4)
        self.relu3 = torch.nn.ReLU()


    def initReformer(self):
        ''''''

        model = ReformerLM(
            num_tokens=20000,
            dim=1024,
            #depth=12,
            depth=4,
            max_seq_len=8192,
            #max_seq_len = 8500,
            #heads=8,
            heads=4,
            lsh_dropout=0.1,
            ff_dropout=0.1,
            post_attn_dropout=0.1,
            layer_dropout=0.1,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
            causal=True,  # auto-regressive or not
            bucket_size=64,  # average size of qk per bucket, 64 was recommended in paper
            n_hashes=4,  # 4 is permissible per author, 8 is the best but slower
            emb_dim=128,  # embedding factorization for further memory savings
            dim_head=64,
            # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
            ff_chunks=200,  # number of chunks for feedforward layer, make higher if there are memory issues
            attn_chunks=8,  # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
            num_mem_kv=128,  # persistent learned memory key values, from all-attention paper
            full_attn_thres=1024,  # use full attention if context length is less than set value
            reverse_thres=1024,    # turn off reversibility for 2x speed for sequence lengths shorter or equal to the designated value
            use_scale_norm=False,  # use scale norm from 'Transformers without tears' paper
            use_rezero=False,      # remove normalization and use rezero from 'ReZero is All You Need'
            #use_rezero=True,      # remove normalization and use rezero from 'ReZero is All You Need'
            #one_value_head=False,  # use one set of values for all heads from 'One Write-Head Is All You Need'
            one_value_head=True,  # use one set of values for all heads from 'One Write-Head Is All You Need'
            weight_tie=False,      # tie parameters of each layer for no memory per additional depth
            weight_tie_embedding=False,
            # use token embedding for projection of output, some papers report better results
            n_local_attn_heads=2,
            # many papers suggest mixing local attention heads aids specialization and improves on certain tasks
            pkm_layers=(4, 7),
            # specify layers to use product key memory. paper shows 1 or 2 modules near the middle of the transformer is best
            pkm_num_keys=128,  # defaults to 128, but can be increased to 256 or 512 as memory allows
            use_full_attn=False
            # only turn on this flag to override and turn on full attention for all sequence lengths. for comparison with LSH to show that it is working
        ).cuda()

        reformer_ap = Autopadder(model)
        return reformer_ap

    def initReformer0(self):
        ''''''
        DE_SEQ_LEN  = 8192
        encoder = ReformerLM(
            num_tokens=20000,
            emb_dim=128,
            dim=1024,
            depth=12,
            heads=8,
            max_seq_len=DE_SEQ_LEN,
            fixed_position_emb=True,
            return_embeddings=True  # return output of last attention layer
        ).cuda()

        reformer_ap = Autopadder(encoder)
        return reformer_ap

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

        #x = self.lstm(x_in)[0]
        x_in = x_in[:, :self.maxlen]
        x = self.forward_reformer(x_in)

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

    def forward_reformer(self, x_in):
        m_factor, d_factor = 10, 700
        x_in = (x_in+d_factor)*m_factor
        x_in = x_in.type(torch.int32)

        x_out, y = torch.zeros(x_in.shape[0], self.reformer_out_channels).to(x_in.device), None
        n = x_in.shape[0]
        tm_st = time.time()
        for i in range(0, n):
            try:
                #x = x_in[i].unsqueeze(0)
                with torch.no_grad():
                   x = x_in[i].unsqueeze(0)# (1, 8192, 20000)
                   y = self.reformer_ap(x)
                   x_out[i]  = torch.mean(y, dim=2) [:, :self.reformer_out_channels]  # (1, 20000)
            except Exception as e:
                traceback.print_exc()
                print(e)

        tm_end = time.time()
        log = "Reformer: Time:{}".format (tm_end-tm_st)
        print(log)

        x_out = x_out.type(torch.float32)/m_factor
        return x_out


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
        lbl = "SageReformerLSTMRawPluginAgeGenderHandedOneHotAtt"
        return lbl
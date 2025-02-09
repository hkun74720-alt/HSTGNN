from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pywt
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
      

        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return h

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # attention——> [B, N, N]
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 3)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_in, n_out, dropout, alpha, nheads, order=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.order = order

        self.attentions = [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for k in range(2, self.order + 1):
            self.attentions_2 = ModuleList(
                [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                 range(nheads)])

        self.out_att = GraphAttentionLayer(n_out * nheads * order, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        for k in range(2, self.order + 1):
            x2 = torch.cat([att(x, adj) for att in self.attentions_2], dim=-1)
            x = torch.cat([x, x2], dim=-1)
        x = F.elu(self.out_att(x, adj))
        return x    
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input




class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        self.dims = 6
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, self.dims))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, self.dims))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):

        day_emb = x[..., 1]  
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]  
        time_day = time_day.transpose(1, 2)#.unsqueeze(-1)

        week_emb = x[..., 2]  
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]  
        time_week = time_week.transpose(1, 2)#.unsqueeze(-1)

        return time_day, time_week


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()

        self.conv1 = nn.Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        # nn.init.xavier_uniform_(self.b)
        self.bn = nn.BatchNorm1d(tem_size)

    def forward(self, seq):

        seq = seq.transpose(3, 2)

        seq = seq.permute(0, 1, 3, 2).contiguous()
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(axis=1)  # b,c,n  [50, 1, 12]

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()

        logits = self.bn(logits).permute(0, 2, 1).contiguous()

        coefs = torch.softmax(logits, -1)
        T_coef = coefs.transpose(-1, -2)

        x_1 = torch.einsum('bcnl,blq->bcnq', seq, T_coef)

        return x_1


class DualChannelLearner(nn.Module):
    def __init__(self, features=128, layers=4, length=12, num_nodes=170, dropout=0.1):
        super(DualChannelLearner, self).__init__()


        self.low_freq_layers = nn.ModuleList([
            TATT_1(features, num_nodes, length) for _ in range(layers)
        ])

        kernel_size = int(length / layers + 1)
        self.high_freq_layers = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(features, features, (1, kernel_size)),
            nn.ReLU(),
            nn.Dropout(dropout)) for _ in range(layers)
        ])

        self.alpha = nn.Parameter(torch.tensor(-5.0))

    def forward(self, XL, XH):
        res_xl = XL
        res_xh = XH

        for layer in self.low_freq_layers:
            XL = layer(XL)
        
        XL = (res_xl[..., -1] + XL[..., -1]).unsqueeze(-1)

        XH = nn.functional.pad(XH, (1, 0, 0, 0))  
        

        for layer in self.high_freq_layers:
            XH = layer(XH)  

        XH = (res_xh[..., -1] + XH[..., -1]).unsqueeze(-1)

        alpha_sigmoid = torch.sigmoid(self.alpha)
        output = alpha_sigmoid * XL + (1 - alpha_sigmoid) * XH

        return output 
 
class HGL_layer(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(HGL_layer, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  # We assume d_v always equals d_k
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        

        #self.attention = SpatialAttention(factor=5, scale=None, attention_dropout=0.1, num_nodes=self.num_nodes)
        self.gcn = gcn(256, 256, dropout, support_len=1, order=1)
        self.gat = GAT(256, 256, dropout, alpha=0.2, nheads=1)

    
        
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

        self.alpha = nn.Parameter(torch.tensor(-5.0)) 
        self.weight = nn.Parameter(torch.ones(256, self.num_nodes, 1))   
        self.bias = nn.Parameter(torch.zeros(256, self.num_nodes, 1))
        
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 6).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(6, num_nodes).to(device), requires_grad=True).to(device)
        

    def forward(self, input, D_Graph):
        #print('input', input.shape)        #input torch.Size([64, 256, 170, 1])        

        A_graph = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1).unsqueeze(0)
        x_gcn = self.gcn(input, [A_graph])

        x_gat = self.gat(input.transpose(1,3), D_Graph).transpose(1,3)


        alpha_sigmoid = torch.sigmoid(self.alpha)  
        x =  alpha_sigmoid* x_gat +  (1 - alpha_sigmoid) * x_gcn
        
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * self.weight + self.bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        

        return x
    
    
class HybridGraphLearner(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length, dropout, num_layers):
        super(HybridGraphLearner, self).__init__()
        

        self.layers = nn.ModuleList([
            HGL_layer(device, 
                    d_model=d_model, 
                    head=head, 
                    num_nodes=num_nodes, 
                    seq_length=seq_length, 
                    dropout=dropout)
            for _ in range(num_layers)  
        ])

    def forward(self, x, D_Graph):

        for layer in self.layers:
            x = layer(x, D_Graph)
        return x

        



class HSTGNN(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=883,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 1
        self.layers = 3
        self.dims = 6

        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358  or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes>200:
            time = 96

        self.Temb = TemporalEmbedding(time, channels)
        self.start_conv_res = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1)) 
        self.start_conv_1 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))  
        self.start_conv_2 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))  
        self.network_channel = channels * 2
        
        self.DCL = DualChannelLearner(
            features = self.node_dim, 
            layers = self.layers, 
            length = self.input_len, 
            num_nodes = self.num_nodes, 
            dropout=0.1
        )
        
        
        self.HGL = HybridGraphLearner(
            device,
            d_model = self.network_channel,
            head = self.head,
            num_nodes = num_nodes,
            seq_length = 1,
            dropout = dropout,
            num_layers = self.layers
        )
        
        
        self.MLP = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.dims,
            kernel_size=(1, 1)
        )
        

        self.E_s = nn.Parameter(torch.randn(self.dims, num_nodes).to(device), requires_grad=True).to(device)

        
        self.fc_st = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )

        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )
        
        
        

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        
        res = history_data
        input_data = history_data
        residual_cpu = input_data.cpu()
        residual_numpy = residual_cpu.detach().numpy()
        coef = pywt.wavedec(residual_numpy, 'db1', level=2)
        coefl = [coef[0]] + [None] * (len(coef) - 1)
        coefh = [None] + coef[1:]
        xl = pywt.waverec(coefl, 'db1')
        xh = pywt.waverec(coefh, 'db1')

        xl = torch.from_numpy(xl).to(self.device)
        xh = torch.from_numpy(xh).to(self.device)

        input_data_1 = self.start_conv_1( xl)  
        input_data_2 = self.start_conv_2( xh)  
        res = self.start_conv_res( res)[..., -1].unsqueeze( -1)


        input_data = self.DCL(input_data_1, input_data_2)
        
        
        E_d = torch.tanh(self.MLP(history_data)[-1, ..., -1] * 
                    (self.Temb(history_data.permute(0, 3, 2, 1))[0] * 
                     self.Temb(history_data.permute(0, 3, 2, 1))[1])[-1, ...] * 
                     self.E_s)


        D_graph = F.softmax(F.relu(torch.mm(E_d.transpose(0,1), E_d)), dim=1)


        data_st = torch.cat([input_data] + [res], dim=1)

        skip = self.fc_st(data_st)
        data_st = self.HGL(data_st, D_graph) + skip


        prediction = self.regression_layer(data_st)

        return prediction

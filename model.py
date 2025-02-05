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

def disentangle(x, w, j):
    x = x.transpose(0,3,2,1)  # [S,D,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    
    xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    xh = pywt.waverec(coefh, w).transpose(0,3,2,1)
    
    return xl[:,:3,:,:], xh[:,:3,:,:]  # 添加返回语句


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
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, 128))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):

        day_emb = x[..., 1]  
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]  
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]  
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]  
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb


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

        # 低频通道：TATT_1 处理 XL
        self.low_freq_layers = nn.ModuleList([
            TATT_1(features, num_nodes, length) for _ in range(layers)
        ])

        # 高频通道：TConv 逻辑合并
        high_freq_layers = []
        kernel_size = int(length / layers + 1)
        for _ in range(layers):
            high_freq_layers.append(nn.Sequential(
                nn.Conv2d(features, features, (1, kernel_size)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        self.high_freq_layers = nn.Sequential(*high_freq_layers)

        # 低频 & 高频融合权重
        self.alpha = nn.Parameter(torch.tensor(-5.0))

    def forward(self, XL, XH):
        # 低频数据通过 TATT_1 逐层处理
        for layer in self.low_freq_layers:
            XL = layer(XL)

        # 高频数据逐层处理
        XH = nn.functional.pad(XH, (1, 0, 0, 0))  # 处理边界
        XH = self.high_freq_layers(XH) + XH[..., -1].unsqueeze(-1)

        # 计算融合权重
        alpha_sigmoid = torch.sigmoid(self.alpha)
        output = alpha_sigmoid * XL[..., -1].unsqueeze(-1) + (1 - alpha_sigmoid) * XH

        return output  # 返回融合后的结果
   
class Spatial_block(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(Spatial_block, self).__init__()
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

        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1).unsqueeze(0)
        x_gcn = self.gcn(input, [adp])

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
        self.blocks = 4

        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358  or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes>200:
            time = 96

        self.Temb = TemporalEmbedding(time, channels)

        

        self.start_conv_1 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))  
        self.start_conv_2 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))  

        self.network_channel = channels * 2


        
        self.DCL = DualChannelLearner()

        self.alpha_s = nn.Parameter(torch.tensor(-5.0)) 

        self.use_emb = True
        self.linear = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        
        
        self.SpatialBlock = Spatial_block(
            device,
            d_model=self.network_channel,
            head=self.head,
            num_nodes=num_nodes,
            seq_length=1,
            dropout=dropout,
        )

        self.fc_st = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )

        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )
        
        
        self.MLP = nn.Conv2d(in_channels=channels,
                                                   out_channels=6,
                                                   kernel_size=(1, 1))

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        #print('history_data', history_data.shape)                #history_data torch.Size([64, 3, 307, 12])

        
        input_data = history_data
        
        #decoupling layer
        residual_cpu = input_data.cpu()
        xl, xh = disentangle(residual_cpu.detach().numpy(), 'db1', 2)
        xl = torch.from_numpy(xl).to('cuda:0')   #low_freq
        xh = torch.from_numpy(xh).to('cuda:0')   #high_freq
        
        #start_conv layer
        history_data = history_data.permute(0, 3, 2, 1)
        input_data_1 = self.start_conv_1( xl)  
        input_data_2 = self.start_conv_2( xh)  #torch.Size([64, 128, 170, 12])
        
        #dual-channel learner
        input_data = self.DCL(input_data_1, input_data_2)
        

        tem_emb = self.Temb(history_data)
        E_st = tem_emb
        E_st = E_st* input_data 
        E_d = self.MLP(E_st).squeeze(-1).transpose(1,2)[-1,:,:]
        
        #dynamic graph construction
        D_graph = F.softmax(F.relu(torch.mm(E_d, E_d.transpose(0,1))), dim=1)

                            
        data_st = torch.cat([input_data] + [tem_emb], dim=1)

        #hybrid grapph learning moodule
        data_st = self.SpatialBlock(data_st, D_graph) + self.fc_st(data_st)

        #output layer
        prediction = self.regression_layer(data_st)

        return prediction

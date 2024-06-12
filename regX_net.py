import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torchvision import datasets
from torcheval.metrics.functional import multiclass_f1_score
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax

import torch
from torch.nn import Parameter, functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes


class subNet(nn.Module):
    def __init__(self, num_peak, num_tf):
        super(subNet, self).__init__()
        self.num_peak = num_peak
        self.num_tf = num_tf

        self.fc1 = nn.Linear(self.num_peak * self.num_tf, 1)
        self.fc1_activate = nn.ReLU()
        self.abs = nn.ReLU()
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

class Net_PPI(nn.Module):
    def __init__(self, num_genes, num_peaks, num_tf, cut):
        super(Net_PPI, self).__init__()
        self.num_peaks = num_peaks
        self.num_tf = num_tf
        self.num_genes = num_genes
        self.cut = cut
        self.gene_dim = 2
               
        self.subnet_modules = nn.ModuleList()
        for i in range(num_genes):
            num_peak = self.num_peaks[i]
            self.subnet = subNet(num_peak, self.num_tf)
            self.subnet_modules.append(self.subnet)
                    
        self.cat_activate = nn.ReLU()
        self.conv = GCNConv(1, self.gene_dim, add_self_loops=False)
        self.conv_activate = nn.ReLU()
        self.out = nn.Linear(self.num_genes*self.gene_dim, 3)
        
        
    def initialize_parameters(self):
        weight = self.conv.lin.weight
        bias = self.conv.bias
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        #torch.nn.init.xavier_uniform_(weight)
        if bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        x_cat = torch.zeros(x.shape[0], 0).to(device)
        for i in range(len(self.subnet_modules)):
            x_sub = x[:, :, self.cut[i]:self.cut[i+1]]
            x_sub = self.subnet_modules[i](x_sub)
            x_cat = torch.cat((x_cat, x_sub), dim=1)

        x_cat = self.cat_activate(x_cat)
        x = torch.unsqueeze(x_cat, 2)
        x = self.conv_activate(self.conv(x, edge))
        x = x.reshape(x.shape[0], -1)
        out = self.out(x)
        return x_cat, x, out

class Net_GO(nn.Module):
    def __init__(self, genes, num_peaks, num_tf, cut, mask, num_gos):
        super(Net_GO, self).__init__()
        self.num_peaks = num_peaks
        self.num_tf = num_tf
        self.genes = genes
        self.cut = cut
        self.mask = mask
        self.num_gos = num_gos
        self.go_dim=6
               
        self.subnet_modules = nn.ModuleList()
        for i in range(len(self.genes)):
            gene = self.genes[i]
            num_peak = self.num_peaks[i]
            self.subnet = subNet(num_peak, self.num_tf)
            self.subnet_modules.append(self.subnet)
                    
        self.cat_activate = nn.LeakyReLU()
        self.fc1_weight = nn.Parameter(torch.Tensor(self.num_gos*self.go_dim, len(self.genes)), requires_grad=True)
        self.mask_rep = nn.Parameter(torch.repeat_interleave(mask, self.go_dim, dim=0), requires_grad=False)
        self.fc1_bias = nn.Parameter(torch.Tensor(self.num_gos*self.go_dim), requires_grad=True)
        self.fc1_activate = nn.LeakyReLU()
        #self.fc1_drop = nn.Dropout()
        self.gat = selfGATConv(self.go_dim, 4, heads=2)
        self.gat_activate = nn.LeakyReLU()
        self.gat_drop = nn.Dropout()
        self.out = nn.Linear(self.num_gos*4, 7)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc1_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.fc1_bias, -bound, bound)
        
    def forward(self, x):
        x_cat = torch.zeros(x.shape[0], 0).to(device)
        for i in range(len(self.subnet_modules)):
            x_sub = x[:, :, self.cut[i]:self.cut[i+1]]
            x_sub = self.subnet_modules[i](x_sub)
            x_cat = torch.cat((x_cat, x_sub), dim=1)

        x_cat = self.cat_activate(x_cat)
        x = x_cat.matmul((self.fc1_weight*self.mask_rep).t()) + self.fc1_bias
        x = self.fc1_activate(x)
        #x = self.fc1_drop(x)
        x = x.reshape(x.shape[0], self.go_dim, -1)
        x = x.permute(2, 0, 1)
        x = self.gat(x, edge)
        x = x.permute(1, 2, 0)
        x = x.reshape(x.shape[0], -1) 
        x = self.gat_activate(x)
        x = self.gat_drop(x)
        out = self.out(x)
        return x_cat, x, out

    class selfGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=2, concat=False,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(selfGATConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.node_dim = 0

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, 1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        # if size is None and torch.is_tensor(x):
        #     edge_index, _ = remove_self_loops(edge_index)
        #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, ptr, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(edge_index_i.shape[0], -1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(edge_index_i.shape[0], -1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            #print((torch.cat([x_i, x_j], dim=-1) * self.att).shape)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr, size_i)
        self.alpha = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # print(alpha.shape)

        return x_j * alpha.view(edge_index_i.shape[0], -1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=2)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



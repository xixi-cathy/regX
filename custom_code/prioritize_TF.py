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
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import scipy
from scipy import stats
import h5py
import argparse
import os
from pyfaidx import Fasta
import math
import sys
from typing import Tuple
from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase, DeepLift, IntegratedGradients, GradientShap, NoiseTunnel, FeatureAblation, KernelShap
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
import scipy.stats as ss
import matplotlib.pyplot as plt
import networkx as nx


class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

class subNet(nn.Module):
    '''
    This class defines the sub-network of a gene.
    '''
    def __init__(self, num_peak, num_tf):
        '''
        num_peak: number of cCREs surrounding this gene.
        num_tf: number of TFs.
        '''
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

class Net_GCN(nn.Module):
    '''
    This class defines the regX network with a GCN layer (as used in the T2D example in the original papaer). 
    Users can choose this when they have an undirected graph describing the relationships among genes (e.g., PPI).
    '''
    def __init__(self, num_genes, num_peaks, num_tf, cut, num_output):
        '''
        num_genes: number of target genes in the hidden layer.
        num_peaks: a list or array of the number of cCREs surrounding all target genes. The length should be the same as num_genes.
        num_tf: number of TFs.
        cut: a list or array of indices to cut the input matrix in the cCRE dimension to split the TAM of different genes. The list should start with 0 and end with the last index to cut the TAM of the last gene. Thus, the length should be num_genes+1.
        num_output: number of output channels (e.g., cell states, classes)
        '''
        super(Net_GCN, self).__init__()
        self.num_peaks = num_peaks
        self.num_tf = num_tf
        self.num_genes = num_genes
        self.cut = cut
        self.gene_dim = 2
        self.num_output = num_output
               
        self.subnet_modules = nn.ModuleList()
        for i in range(num_genes):
            num_peak = self.num_peaks[i]
            self.subnet = subNet(num_peak, self.num_tf)
            self.subnet_modules.append(self.subnet)
                    
        self.cat_activate = nn.ReLU()
        self.conv = GCNConv(1, self.gene_dim, add_self_loops=False)
        self.conv_activate = nn.ReLU()
        self.out = nn.Linear(self.num_genes*self.gene_dim, self.num_output)
        self.initialize_parameters()
        
    def initialize_parameters(self):
        weight = self.conv.lin.weight
        nn.init.constant_(weight[0], 2*torch.rand(1)[0]+0.1)
        nn.init.uniform_(weight[1], -2, 2)
                

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
        return out

class Net_SAGEConv(nn.Module):
    '''
    This class defines the regX network with a SAGEGonv layer. 
    Users can choose this when they have an undirected graph describing the relationships among genes (e.g., PPI).
    '''
    def __init__(self, num_genes, num_peaks, num_tf, cut, num_output):
        '''
        num_genes: number of target genes in the hidden layer.
        num_peaks: a list or array of the number of cCREs surrounding all target genes. The length should be the same as num_genes.
        num_tf: number of TFs.
        cut: a list or array of indices to cut the input matrix in the cCRE dimension to split the TAM of different genes. The list should start with 0 and end with the last index to cut the TAM of the last gene. Thus, the length should be num_genes+1.
        num_output: number of output channels (e.g., cell states, classes)
        '''
        super(Net_SAGEConv, self).__init__()
        self.num_peaks = num_peaks
        self.num_tf = num_tf
        self.num_genes = num_genes
        self.cut = cut
        self.gene_dim = 2
        self.num_output = num_output
               
        self.subnet_modules = nn.ModuleList()
        for i in range(num_genes):
            num_peak = self.num_peaks[i]
            self.subnet = subNet(num_peak, self.num_tf)
            self.subnet_modules.append(self.subnet)
                    
        self.cat_activate = nn.ReLU()
        self.conv = SAGEConv(1, self.gene_dim, add_self_loops=False)
        self.conv_activate = nn.ReLU()
        self.out = nn.Linear(self.num_genes*self.gene_dim, self.num_output)
        
        self.initialize_parameters()
        
    def initialize_parameters(self):
        weight_l = self.conv.lin_l.weight
        nn.init.constant_(weight_l[0][0], 2*torch.rand(1)[0]+0.1)
        nn.init.uniform_(weight_l[1][0], -2, 2)
        weight_r = self.conv.lin_r.weight
        nn.init.constant_(weight_r[0][0], 2*torch.rand(1)[0]+0.1)
        nn.init.uniform_(weight_r[1][0], -2, 2)

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
        return out    
    
    
class Net_GAT(nn.Module):
    '''
    This class defines the regX network with a GAT layer (as used in the hair follicle example in the original papaer). 
    Users can choose this when they have a directed acyclic graph describing the relationships among genes (e.g., GO).
    '''
    def __init__(self, genes, num_peaks, num_tf, cut, mask, num_gos, num_output):
        '''
        genes: a list of target genes in the hidden layer.
        num_peaks: a list or array of the number of cCREs surrounding all target genes. The length should be the same as that of genes.
        num_tf: number of TFs.
        cut: a list or array of indices to cut the input matrix in the cCRE dimension to split the TAM of different genes. The list should start with 0 and end with the last index to cut the TAM of the last gene. Thus, the length should be len(genes)+1.
        mask: a GO-by-gene binary matrix describing whether the gene is related to the GO term (1) or not (0). This matrix can be generated from the GO annotation file.
        num_gos: number of GO terms included.
        num_output: number of output channels (e.g., cell states, classes)
        '''
        super(Net_GAT, self).__init__()
        self.num_peaks = num_peaks
        self.num_tf = num_tf
        self.genes = genes
        self.cut = cut
        self.mask = mask
        self.num_gos = num_gos
        self.go_dim=6
        self.num_output = num_output
               
        self.subnet_modules = nn.ModuleList()
        for i in range(len(self.genes)):
            #gene = self.genes[i]
            num_peak = self.num_peaks[i]
            self.subnet = subNet(num_peak, self.num_tf)
            self.subnet_modules.append(self.subnet)
                    
        self.cat_activate = nn.LeakyReLU()
        self.fc1_weight = nn.Parameter(torch.Tensor(self.num_gos*self.go_dim, len(self.genes)), requires_grad=True)
        self.mask_rep = nn.Parameter(torch.repeat_interleave(mask, self.go_dim, dim=0), requires_grad=False)
        self.fc1_bias = nn.Parameter(torch.Tensor(self.num_gos*self.go_dim), requires_grad=True)
        self.fc1_activate = nn.LeakyReLU()
        self.gat = selfGATConv(self.go_dim, 4, heads=2)
        self.gat_activate = nn.LeakyReLU()
        self.gat_drop = nn.Dropout()
        self.out = nn.Linear(self.num_gos*4, self.num_output)
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
        x = x.reshape(x.shape[0], self.go_dim, -1)
        x = x.permute(2, 0, 1)
        x = self.gat(x, edge)
        x = x.permute(1, 2, 0)
        x = x.reshape(x.shape[0], -1) 
        x = self.gat_activate(x)
        x = self.gat_drop(x)
        out = self.out(x)
        return out


class selfGATConv(MessagePassing):
    '''
    This class is a self-defined GAT layer.
    '''
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

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr, size_i)
        self.alpha = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

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

    
parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, default='./', help='Path of the data files (all data files should be saved under a same directory)')
parser.add_argument('--savepath', type=str, default='./', help='Path of the result folder')
parser.add_argument('--ref', type=str, default='hg19', help='The reference genome. Currently, we only support hg19 and mm10. You may modify the source code and download relevant files if using other reference genomes.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for training. cpu or cuda:2, etc.')
parser.add_argument('--model', type=str, default='GCN', help='GNN type. GCN, GAT or SAGEConv')
parser.add_argument('--batchsize', type=int, default=256, help='Batchsize when training regX')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate when training regX')
parser.add_argument('--top', type=int, default=10, help='Number of driver TFs between each two states in the output')

args = parser.parse_args()
    
os.chdir(args.filepath)   

if args.ref=="hg19":
    filename = "JASPAR_motifs_pfm_homosapiens"
elif args.ref=="mm10":
    filename = "JASPAR_motifs_pfm_mouse"
else:
    sys.exit("The provided reference genome version not supported. Currently, we only support hg19 and mm10. For other usage, please modify the source code.")
    

### Loading data
print('\nLoading data...')
df_x = pd.read_csv('./atac.csv', index_col=0).transpose()
df_y = pd.read_csv('./rna.csv', index_col=0).transpose()
df_peaks = pd.DataFrame(df_x.columns)[0].str.split('-',expand=True) 
df_peaks = df_peaks.rename(columns={0: "chrom", 1: "start", 2: "end"})
df_peaks["start"] = pd.to_numeric(df_peaks["start"])
df_peaks["end"] = pd.to_numeric(df_peaks["end"])

motif_files = os.listdir('./' + filename + '/pfm.np/')
tfs_kept = []
for i in list(motif_files):
    if args.ref=="hg19":
        tf = i.split('.')[-2]
    else:
        tf = i.split('.')[-2].capitalize()
    if tf in df_y.columns:
        if tf in tfs_kept:
            continue
            
        tfs_kept.append(tf)
tfs_kept = sorted(tfs_kept)

h5f = h5py.File(args.savepath+'processing/data.h5', 'r')
X = h5f['X'][:]
num_peaks = h5f['num_peaks'][:]
cell_by_tf = h5f['cell_by_tf'][:]
cell_by_peak = h5f['cell_by_peak'][:]
W = h5f['W'][:]
h5f.close()

cut = [0]
s = 0
for i in num_peaks:
    s = s+i
    cut.append(s)

device = torch.device(args.device)

if args.model=='GCN' or args.model=='SAGEConv':
    edge = pd.read_csv(args.savepath+'processing/ppi_filtered.txt', sep='\t')
    edge = torch.cat((torch.from_numpy(np.array(edge)).t(), torch.from_numpy(np.array(edge)).t().flip([0])), dim=1).to(device)
elif args.model=='GAT':
    go = pd.read_csv('./go_filtered.txt', sep='\t')
    goa = pd.read_csv('./goa_filtered.txt', sep='\t')
    genes_filtered = pd.read_csv(args.savepath+'processing/pearson.testset.txt', sep='\t', header=None, index_col=0).index.to_list()

    goa_filtered = goa.loc[goa['X2'].isin(genes_filtered),]
    gos = goa_filtered['X4'].drop_duplicates().to_list()
    go_filtered = go.loc[go['id1'].isin(gos) & go['id2'].isin(gos),]
    gos = goa_filtered['X4'].drop_duplicates().to_list()
    
    mask = torch.zeros(len(genes_filtered), len(gos))
    for i in range(len(goa_filtered['X2'])):
        if goa_filtered['X2'].iloc[i] in genes_filtered:
            mask[genes_filtered.index(goa_filtered['X2'].iloc[i]), gos.index(goa_filtered['X4'].iloc[i])] = 1
    mask = mask.t()
    
    W_true = np.zeros((len(gos), len(gos)))
    for i in range(len(go_filtered)):
        W_true[gos.index(go_filtered['id1'].iloc[i]), gos.index(go_filtered['id2'].iloc[i])]=1
        
    edge_x = []
    edge_y = []
    for i in range(len(go_filtered)):
        edge_x.append(gos.index(go_filtered['id1'].iloc[i]))
        edge_y.append(gos.index(go_filtered['id2'].iloc[i]))
    edge_x = np.array(edge_x)
    edge_y = np.array(edge_y)
    edge = np.vstack([edge_x, edge_y])
    edge = torch.from_numpy(edge).to(device)
else:
    sys.exit("Assigned GNN type not supported. For other usage, please modify the source code.")
    
print("Done!")

print("\nOriginal labels and corresponding channels:")
y = pd.read_csv('./label.csv')
y = np.array(y.iloc[:, 0].to_list())
y = y.reshape((len(y), 1))
enc = OneHotEncoder(handle_unknown='ignore')
y_oht = enc.fit_transform(y).toarray()
for i, category in enumerate(enc.categories_[0]):
    print(f"{category}: {i}")

X_train, X_test, y_oht_train, y_oht_test, y_train, y_test = train_test_split(X, y_oht, y, test_size=0.3,random_state=2024, stratify=y)
X_test, X_val, y_oht_test, y_oht_val, y_test, y_val = train_test_split(X_test, y_oht_test, y_test, test_size=0.5, random_state=2024, stratify=y_test)
indices_train, indices_test, y_label_train, y_label_test= train_test_split(np.arange(X.shape[0]), y, test_size=0.3, random_state=2024, stratify=y)
indices_test, indices_val, y_label_test, y_label_val= train_test_split(indices_test, y_label_test, test_size=0.5, random_state=2024, stratify=y_label_test)

X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_oht_test).to(device)

if os.path.exists(args.savepath+'processing/'+args.model+'_TF_up.npy'):
    print("\nin silico upregulation of TFs have been performed.")
    model_files = [f for f in os.listdir(args.savepath + '/results/model_'+args.model+'/') if f.startswith('model_'+str(args.batchsize)+'_'+str(args.lr))]
    repeats = len(model_files)
    results = np.load(args.savepath+'processing/'+args.model+'_TF_up.npy')
else:
    print("\nPerforming in silico upregulation of TFs...")
    model_files = [f for f in os.listdir(args.savepath + '/results/model_'+args.model+'/') if f.startswith('model_'+str(args.batchsize)+'_'+str(args.lr))]
    repeats = len(model_files)
    print(str(repeats)+" replicates in total")
    num_clusters = y_oht.shape[1]
    results = np.zeros([repeats, X_test.shape[0], X_test.shape[1], num_clusters])
    results_orig = np.zeros([repeats, X_test.shape[0], X_test.shape[1], num_clusters])
    results_perturb = np.zeros([repeats, X_test.shape[0], X_test.shape[1], num_clusters])
    for i in range(repeats):
        print('Calculating replicate '+str(i+1)+":")
        if args.model=="GCN":
            model = Net_GCN(num_genes=len(num_peaks), num_peaks=num_peaks, num_tf=X.shape[1], cut=cut, num_output=y_oht.shape[1]).to(device)
        if args.model=="SAGEConv":
            model = Net_SAGEConv(num_genes=len(num_peaks), num_peaks=num_peaks, num_tf=X.shape[1], cut=cut, num_output=y_oht.shape[1]).to(device)
        if args.model=="GAT":
            model = Net_GAT(genes=genes_filtered, num_peaks=num_peaks, num_tf=X.shape[1], cut=cut, mask=mask, num_gos=W_true.shape[0], num_output=y_oht.shape[1]).to(device)
        model.eval()
        model.load_state_dict(torch.load(args.savepath + '/results/model_'+args.model+'/model_'+str(args.batchsize)+'_'+str(args.lr)+'_rep'+str(i+1)+'.pt', 
                                        map_location=device))
        np.random.seed(123)  

        orig = model(X_test)
        orig = F.softmax(orig)
        for j in range(X_test.shape[1]):
            X_new = torch.clone(X_test).detach()

            tf_mat = cell_by_tf[indices_test, j] + np.quantile(cell_by_tf[:, j], 0.9)
            tf_mat = np.repeat(tf_mat[:, np.newaxis], X_test.shape[2], axis=1)
            peak_mat = cell_by_peak[indices_test, :]
            w = W[j, :]
            w = np.repeat(w[np.newaxis, :], X_test.shape[0], axis=0)
            new = tf_mat*peak_mat*w

            X_new[:, j, :] = torch.from_numpy(new).float().to(device)
            perturb = model(X_new)
            perturb = F.softmax(perturb)
            results[i, :, j, :] = (perturb-orig).detach().cpu().numpy()
            results_orig[i, :, j, :] = orig.clone().detach().cpu().numpy()
            results_perturb[i, :, j, :] = perturb.detach().cpu().numpy()
        print("Done!")

    np.save(args.savepath+'processing/'+args.model+'_TF_up.npy', results)
    np.save(args.savepath+'processing/'+args.model+'_TF_up_orig.npy', results_orig)
    np.save(args.savepath+'processing/'+args.model+'_TF_up_perturb.npy', results_perturb)


print("\nRanking TFs...")
cls = {}
states = []
for i, category in enumerate(enc.categories_[0]):
    cls[i] = category
    states.append(category)
start = []
end = []
for i in range(len(states)):
    for j in range(len(states)):
        if j!=i:
            start.append(states[i])
            end.append(states[j])            
y_label_test = y_label_test.flatten()
top_tfs = []
for j in range(len(start)):
    rank_all = []
    for rep in range(repeats):
        cells = results[rep, np.array(y_label_test)==start[j], :, cls[end[j]]]
        rank = ss.rankdata(np.array([np.sum(cells[:, i]) for i in range(len(tfs_kept))]))
        rank_all.append(rank)

    rank_all = np.array(rank_all)
    top = args.top
    top_tfs_ind = np.argsort(np.mean(rank_all, axis=0))[::-1][:top]
    top_over = [tfs_kept[i] for i in top_tfs_ind]
    top_tfs.append(top_over)
df_results = pd.DataFrame(top_tfs).transpose()
df_results.columns = [str(start[j])+' -> '+str(end[j]) for j in range(len(start))]
df_results.to_csv(args.savepath+'results/'+args.model+'.TF_rank_upregulation.csv', index=False)
print("Done!")


print("Drawing transition plot for top "+str(top)+" TFs between each twe states...")
tfs_draw = pd.DataFrame([i for j in top_tfs for i in j])[0].drop_duplicates().to_list()
if not os.path.exists(args.savepath+'results/'+args.model+'.TF_upregulation/'):
    os.makedirs(args.savepath+'results/'+args.model+'.TF_upregulation/')
rep = 0
cells = results[rep, :, :, :]
inds = [tfs_kept.index(i) for i in tfs_draw]
for ind in inds:
    mat = np.zeros((y_oht.shape[1], y_oht.shape[1]))
    for i in range(y_oht.shape[1]):
        for j in range(y_oht.shape[1]):
            #print(np.array(y_label_test)==cls[i])
            #print(ind)
            #print(j)
            mat[i, j] = np.mean(cells[np.array(y_label_test)==cls[i], ind, j])

    tf = tfs_kept[ind]

    plt.figure(figsize=(4.2,4.2))
    #plt.gcf().set_size_inches(5, 5)
    G = nx.DiGraph()
    G.add_nodes_from(states)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            G.add_edge(states[i], states[j], weight=mat[i,j])

    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    edge_colors = ['tomato' if weight >= 0 else 'skyblue' for weight in edge_weights]
    width = [abs(weight)*30 for weight in edge_weights]
    node_size=[np.sum(mat[i,:] + mat[:,i]) for i in range(y_oht.shape[1])]

    node_colors = ['gold' for state in G.nodes]

    pos = nx.circular_layout(G, scale=1)
    nx.draw(G, pos, with_labels=True, 
            node_color=node_colors, 
            node_size = [500*i+500 for i in node_size],
            width=width, 
            edge_color=edge_colors, connectionstyle="arc3,rad=0.2",
            font_size=10, font_color='black', alpha=0.9, arrowsize=25)

    plt.suptitle(tf+' (up)', fontweight='bold')
    plt.savefig(args.savepath+'results/'+args.model+'.TF_upregulation/'+tf+'.png')
    plt.show()
    plt.close()
print("Done!")    



if os.path.exists(args.savepath+'processing/'+args.model+'_TF_down.npy'):
    print("\nin silico downregulation of TFs have been performed.")
    model_files = [f for f in os.listdir(args.savepath + '/results/model_'+args.model+'/') if f.startswith('model_'+str(args.batchsize)+'_'+str(args.lr))]
    repeats = len(model_files)
    results = np.load(args.savepath+'processing/'+args.model+'_TF_down.npy')
else:
    print("\nPerforming in silico downregulation of TFs...")
    model_files = [f for f in os.listdir(args.savepath + '/results/model_'+args.model+'/') if f.startswith('model_'+str(args.batchsize)+'_'+str(args.lr))]
    repeats = len(model_files)
    print(str(repeats)+" replicates in total")
    num_clusters = y_oht.shape[1]
    results = np.zeros([repeats, X_test.shape[0], X_test.shape[1], num_clusters])
    results_orig = np.zeros([repeats, X_test.shape[0], X_test.shape[1], num_clusters])
    results_perturb = np.zeros([repeats, X_test.shape[0], X_test.shape[1], num_clusters])
    for i in range(repeats):
        print('Calculating replicate '+str(i+1)+":")
        if args.model=="GCN":
            model = Net_GCN(num_genes=len(num_peaks), num_peaks=num_peaks, num_tf=X.shape[1], cut=cut, num_output=y_oht.shape[1]).to(device)
        if args.model=="SAGEConv":
            model = Net_SAGEConv(num_genes=len(num_peaks), num_peaks=num_peaks, num_tf=X.shape[1], cut=cut, num_output=y_oht.shape[1]).to(device)
        if args.model=="GAT":
            model = Net_GAT(genes=genes_filtered, num_peaks=num_peaks, num_tf=X.shape[1], cut=cut, mask=mask, num_gos=W_true.shape[0], num_output=y_oht.shape[1]).to(device)
        model.eval()
        model.load_state_dict(torch.load(args.savepath + '/results/model_'+args.model+'/model_'+str(args.batchsize)+'_'+str(args.lr)+'_rep'+str(i+1)+'.pt', 
                                        map_location=device))
        np.random.seed(123)  

        orig = model(X_test)
        orig = F.softmax(orig)
        for j in range(X_test.shape[1]):
            X_new = torch.clone(X_test).detach()

            X_new[:, j, :] = 0
            perturb = model(X_new)
            perturb = F.softmax(perturb)
            results[i, :, j, :] = (perturb-orig).detach().cpu().numpy()
            results_orig[i, :, j, :] = orig.clone().detach().cpu().numpy()
            results_perturb[i, :, j, :] = perturb.detach().cpu().numpy()
        print("Done!")

    np.save(args.savepath+'processing/'+args.model+'_TF_down.npy', results)
    np.save(args.savepath+'processing/'+args.model+'_TF_down_orig.npy', results_orig)
    np.save(args.savepath+'processing/'+args.model+'_TF_down_perturb.npy', results_perturb)

print("\nRanking TFs...")
top_tfs = []
for j in range(len(start)):
    rank_all = []
    for rep in range(repeats):
        cells = results[rep, np.array(y_label_test)==start[j], :, cls[end[j]]]
        rank = ss.rankdata(np.array([np.sum(cells[:, i]) for i in range(len(tfs_kept))]))
        rank_all.append(rank)

    rank_all = np.array(rank_all)
    top = args.top
    top_tfs_ind = np.argsort(np.mean(rank_all, axis=0))[::-1][:top]
    top_over = [tfs_kept[i] for i in top_tfs_ind]
    top_tfs.append(top_over)
df_results = pd.DataFrame(top_tfs).transpose()
df_results.columns = [str(start[j])+' -> '+str(end[j]) for j in range(len(start))]
df_results.to_csv(args.savepath+'results/'+args.model+'.TF_rank_downregulation.csv', index=False)
print("Done!")

print("Drawing transition plot for top "+str(top)+" TFs between each twe states...")
tfs_draw = pd.DataFrame([i for j in top_tfs for i in j])[0].drop_duplicates().to_list()
if not os.path.exists(args.savepath+'results/'+args.model+'.TF_downregulation/'):
    os.makedirs(args.savepath+'results/'+args.model+'.TF_downregulation/')
rep = 0
cells = results[rep, :, :, :]
inds = [tfs_kept.index(i) for i in tfs_draw]
for ind in inds:
    mat = np.zeros((y_oht.shape[1], y_oht.shape[1]))
    for i in range(y_oht.shape[1]):
        for j in range(y_oht.shape[1]):
            mat[i, j] = np.mean(cells[np.array(y_label_test)==cls[i], ind, j])

    tf = tfs_kept[ind]

    plt.figure(figsize=(4.2,4.2))
    #plt.gcf().set_size_inches(5, 5)
    G = nx.DiGraph()
    G.add_nodes_from(states)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            G.add_edge(states[i], states[j], weight=mat[i,j])

    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    edge_colors = ['tomato' if weight >= 0 else 'skyblue' for weight in edge_weights]
    width = [abs(weight)*30 for weight in edge_weights]
    node_size=[np.sum(mat[i,:] + mat[:,i]) for i in range(y_oht.shape[1])]

    node_colors = ['gold' for state in G.nodes]

    pos = nx.circular_layout(G, scale=1)
    nx.draw(G, pos, with_labels=True, 
            node_color=node_colors, 
            node_size = [500*i+500 for i in node_size],
            width=width, 
            edge_color=edge_colors, connectionstyle="arc3,rad=0.2",
            font_size=10, font_color='black', alpha=0.9, arrowsize=25)

    plt.suptitle(tf+' (down)', fontweight='bold')
    plt.savefig(args.savepath+'results/'+args.model+'.TF_downregulation/'+tf+'.png')
    plt.show()
    plt.close()
print("Done!") 
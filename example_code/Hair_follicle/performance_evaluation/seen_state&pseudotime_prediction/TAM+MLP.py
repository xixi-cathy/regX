import scanpy as sc
import pandas as pd
import numpy as np
from pyfaidx import Fasta
from verstack import stratified_continuous_split
import matplotlib.pyplot as plt
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

import h5py
import seaborn as sns
import os
os.chdir('/nfs/public/xixi/scRegulate/SHAREseq')

from typing import Tuple

# Data preprocess

df_x = pd.read_csv('atac.aggregate_30cells.csv', index_col=0).transpose()
df_y = pd.read_csv('rna.aggregate_30cells.csv', index_col=0).transpose()
df_peaks = pd.DataFrame(df_x.columns)[0].str.split('-',expand=True) 
df_peaks = df_peaks.rename(columns={0: "chrom", 1: "start", 2: "end"})
df_peaks["start"] = pd.to_numeric(df_peaks["start"])
df_peaks["end"] = pd.to_numeric(df_peaks["end"])
geneanno = pd.read_csv('../../ref_genome/mm10_geneanno.txt', sep='\t')
geneanno = geneanno.drop_duplicates(subset=['Gene name'])

# Prepare data
motif_files = os.listdir('../../ref_genome/JASPAR_motifs_pfm_mouse/pfm.np')

tfs_kept = []
tf_by_region_mat = []
for i in list(motif_files):
    tf = i.split('.')[-2].capitalize()
    if tf in df_y.columns:
        if tf in tfs_kept:
            continue
            
        tfs_kept.append(tf)
tfs_kept = sorted(tfs_kept)

go = pd.read_csv('./go/go_follicle.txt', sep='\t')
goa = pd.read_csv('./go/goa_follicle.txt', sep='\t')

# Data preparation for nn
files = os.listdir('/nfs/public/xixi/scRegulate/SHAREseq/nn.best.feature6.learnW_go')
markers_filtered = []
for file in files:
    marker = file.split('.')[0]
    if marker not in markers_filtered:
        markers_filtered.append(marker)

genes = goa['X2'].drop_duplicates().to_list()
print(len(genes))
genes_filtered = [i for i in genes if i not in tfs_kept]
print(len(genes_filtered))
genes_filtered = [i for i in genes_filtered if i in markers_filtered]
print(len(genes_filtered))

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

h5f = h5py.File('./predict_lineage_pseudotime/data_follicle2.h5', 'r')
X = h5f['X'][:]
expr = h5f['expr'][:]
num_peaks = h5f['num_peaks'][:]
peaks = pd.DataFrame(h5f['num_peaks'][:])
h5f.close()

cut = [0]
s = 0
for i in num_peaks:
    s = s+i
    cut.append(s)

y = pd.read_csv('skin.aggregate.cellid&cluster&pseudotime_30cells.csv', index_col=0)
time = np.array(y['aggr_pseudotime'])
y_std = (time - time.min()) / (time.max() - time.min())
y_pseudo = y_std * (1 - (0)) + (0)
y_pseudo = y_pseudo.reshape((len(y), 1))

y = np.array(y['celltype'])
y_new = y.reshape((len(y), 1))
enc = OneHotEncoder(handle_unknown='ignore')
y_oht = enc.fit_transform(y_new).toarray()
y_oht = np.concatenate([y_oht, y_pseudo], axis=1)

X_train, X_test, expr_train, expr_test, y_oht_train, y_oht_test, y_train, y_test = train_test_split(X, expr, y_oht, y, test_size=0.3,
                                                                                                    random_state=2023, stratify=y)
X_test, X_val, expr_test, expr_val, y_oht_test, y_oht_val, y_test, y_val = train_test_split(X_test, expr_test, y_oht_test, y_test, 
                                                                                                test_size=0.5, random_state=2023, 
                                                                                            stratify=y_test)

X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
X_test = torch.from_numpy(X_test).float()
expr_train = torch.from_numpy(expr_train).float()
expr_val = torch.from_numpy(expr_val).float()
expr_test = torch.from_numpy(expr_test).float()
y_train = torch.from_numpy(y_oht_train).float()
y_val = torch.from_numpy(y_oht_val).float()
y_test = torch.from_numpy(y_oht_test).float()

# Def

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
#from torch_scatter import scatter_max, scatter_add
import math

edge_x = []
edge_y = []
for i in range(len(go_filtered)):
    edge_x.append(gos.index(go_filtered['id1'].iloc[i]))
    edge_y.append(gos.index(go_filtered['id2'].iloc[i]))
edge_x = np.array(edge_x)
edge_y = np.array(edge_y)
edge = np.vstack([edge_x, edge_y])
edge = torch.from_numpy(edge)

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


class Net(nn.Module):
    def __init__(self, genes, num_peaks, num_tf, cut, mask, num_gos):
        super(Net, self).__init__()
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
                    
        self.cat_activate = nn.ReLU()
        self.fc1 = nn.Linear(len(self.genes), self.num_gos*self.go_dim)
        self.fc1_activate = nn.ReLU()
        self.out = nn.Linear(self.num_gos*self.go_dim, 7)
                
    def forward(self, x):
        x_cat = torch.zeros(x.shape[0], 0).to(device)
        for i in range(len(self.subnet_modules)):
            x_sub = x[:, :, self.cut[i]:self.cut[i+1]]
            x_sub = self.subnet_modules[i](x_sub)
            x_cat = torch.cat((x_cat, x_sub), dim=1)

        x_cat = self.cat_activate(x_cat)
        x = self.fc1_activate(self.fc1(x_cat))
        #x = x.reshape(x.shape[0], -1)
        out = self.out(x)
        return x_cat, x, out

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    #train_loss = 0
    for batch_idx, (data, expr, target) in enumerate(train_loader):
        data, expr, target = data.to(device), expr.to(device), target.to(device)
        optimizer.zero_grad()
        expr_hat, cluster_repr, output = model(data)
        loss_out = out_criterion(output[:, :6], target[:, :6])
        loss_pseudo = expr_criterion(output[:, -1], target[:, -1])
        loss_expr = expr_criterion(expr_hat, expr)
        loss = 0.02*loss_out + 1*loss_pseudo
        loss.backward()
        optimizer.step()
        #model.fc1.weight.data = model.fc1.weight.mul(torch.repeat_interleave(mask.to(device), 4, dim=0))
#         if batch_idx % batchsize == 0:
#             print('\nTrain Epoch: {} [{}/{} ({:.0f}%)], Total loss: {:.6f}, Expr loss: {:.6f}, Cluster loss: {:.6f}, Pseudo loss: {:.6f}'.
#                   format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item(), loss_expr.item(), loss_out.item(), loss_pseudo.item()))
        #return(train_loss)

                  
def test(model, device, test_loader, num_clusters):
    model.eval()
    with torch.no_grad():
        outputs = torch.zeros(0, num_clusters).to(device)
        targets = torch.zeros(0).to(device)
        outputs_pseudo = torch.zeros(0).to(device)
        targets_pseudo = torch.zeros(0).to(device)
        test_loss = 0
        for data, expr, target in test_loader:
            data, expr, target = data.to(device), expr.to(device), target.to(device)
            expr_hat, cluster_repr, output = model(data)
            
            loss_out = out_criterion(output[:, :6], target[:, :6])
            loss_pseudo = expr_criterion(output[:, -1], target[:, -1])
            loss_expr = expr_criterion(expr_hat, expr)
            loss = 0.02*loss_out + 1*loss_pseudo
            test_loss = test_loss+loss.item()

            target_cluster = target[:, :6].argmax(dim=1)
            output_cluster = output[:, :6].softmax(dim=1)
            outputs = torch.cat((outputs, output_cluster), dim=0)
            targets = torch.cat((targets, target_cluster), dim=0)
            target_pseudo = target[:, -1]
            output_pseudo = output[:, -1]
            outputs_pseudo = torch.cat((outputs_pseudo, output_pseudo), dim=0)
            targets_pseudo = torch.cat((targets_pseudo, target_pseudo), dim=0)
        f1_score = multiclass_f1_score(outputs, targets, num_classes=num_clusters)
        pearsonr, _ = stats.pearsonr(targets_pseudo.detach().cpu().numpy(), outputs_pseudo.detach().cpu().numpy())

    return(f1_score, pearsonr, test_loss)

def correlation_score(y_true, y_pred):
    #print(np.corrcoef(y_true, y_pred))
    return np.corrcoef(y_true, y_pred)[1, 0]

def spearman_correlation(y_true, y_pred):
    statistic, pvalue = stats.spearmanr(y_true, y_pred)
    return abs(statistic)

def pearson_correlation(y_true, y_pred):
    statistic, pvalue = stats.pearsonr(y_true, y_pred)
    return abs(statistic[0])

# Demo

use_cuda = True
device = torch.device("cuda:2" if use_cuda else "cpu")
batchsize = 256

train_dataset = TensorDataset(X_train, expr_train, y_train)
val_dataset = TensorDataset(X_val, expr_val, y_val)
test_dataset = TensorDataset(X_test, expr_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

edge_x = []
edge_y = []
for i in range(len(go_filtered)):
    edge_x.append(gos.index(go_filtered['id1'].iloc[i]))
    edge_y.append(gos.index(go_filtered['id2'].iloc[i]))
edge_x = np.array(edge_x)
edge_y = np.array(edge_y)
edge = np.vstack([edge_x, edge_y])
edge = torch.from_numpy(edge).to(device)

for rep in range(10):
    model = Net(genes=genes_filtered, num_peaks=num_peaks, num_tf=X.shape[1], cut=cut, mask=mask, num_gos=W_true.shape[0]).to(device)
    out_criterion = nn.CrossEntropyLoss()
    expr_criterion=nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    train_losses = []
    test_losses = []
    val_losses = []
    es = 0
    best_score = 1000
    for epoch in range(1, 1000+1):
        model.train()
        train(model, device, train_loader, optimizer, epoch)

        model.eval()
        train_score, train_pearsonr, train_loss = test(model, device, train_loader, num_clusters=y_oht.shape[1]-1)
        val_score, val_pearsonr, val_loss = test(model, device, val_loader, num_clusters=y_oht.shape[1]-1)
        test_score, test_pearsonr, test_loss = test(model, device, test_loader, num_clusters=y_oht.shape[1]-1)

        train_losses.append(train_loss/X_train.shape[0])
        val_losses.append(val_loss/X_val.shape[0])
        test_losses.append(test_loss/X_test.shape[0])

        if val_loss < best_score:
            best_score = val_loss
            best_test = test_pearsonr
            best_f1 = test_score.item()
            es = 0
            torch.save(model.state_dict(), './predict_lineage_pseudotime/model.MLP.'+str(rep)+'.pt')
        else:
            es += 1
            #print("Counter {} of 100".format(es))

            if es > 100:
                print("Val: ", best_score, "Test: ", best_test, "Test F1: ", best_f1)
                break   

                
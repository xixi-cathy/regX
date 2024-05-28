# This script describes the second step of training, which is to train the entire model from scratch from the learned TAMs in the first step of training.
# We trained the model for 10 times with random seeds, to guarantee the model's robustness in providing reliable biological discoveries.

import scanpy as sc
import pandas as pd
import numpy as np
from pyfaidx import Fasta
from verstack import stratified_continuous_split
import matplotlib.pyplot as plt
import time
import math

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
from torch_geometric.nn import GCNConv

import h5py
import seaborn as sns
import os
os.chdir('/nfs/public/xixi/scRegulate/T2D')

from typing import Tuple

# Data preprocess

df_x = pd.read_csv('./data/beta.atac.aggregate_30cells.csv', index_col=0).transpose()
df_y = pd.read_csv('./data/beta.rna.aggregate_30cells.csv', index_col=0).transpose()
df_peaks = pd.DataFrame(df_x.columns)[0].str.split('-',expand=True) 
df_peaks = df_peaks.rename(columns={0: "chrom", 1: "start", 2: "end"})
df_peaks["start"] = pd.to_numeric(df_peaks["start"])
df_peaks["end"] = pd.to_numeric(df_peaks["end"])

#df_peaks.to_csv('/data1/xixi/scRegulate/multiomic_data/10x_PBMC/peaks_all.csv')

geneanno = pd.read_csv('../../ref_genome/hg19_geneanno.txt', sep='\t')
geneanno = geneanno.drop_duplicates(subset=['Gene name'])
# Prepare data

motif_files = os.listdir('../../ref_genome/JASPAR_motifs_pfm_homosapiens/pfm.np')
tfs_kept = []
tf_by_region_mat = []
for i in list(motif_files):
    tf = i.split('.')[-2]#.capitalize()
    if tf in df_y.columns:
        if tf in tfs_kept:
            continue
            
        tfs_kept.append(tf)
tfs_kept = sorted(tfs_kept)

# Data preparation for nn

h5f = h5py.File('./predict_status/data_T2D_float16.h5', 'r')
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

y = pd.read_csv('./data/beta.label.aggregate_30cells.csv', index_col=0)

y = y['status'].to_list()
y_new = y
values_to_replace = ['Non-diabetic', 'Pre-T2D', 'T2D']
for value in values_to_replace:
    y_new = np.where(y == value, 0, y_new)
y_new = y_new.reshape((len(y), 1))
enc = OneHotEncoder(handle_unknown='ignore')
y_oht = enc.fit_transform(y_new).toarray()
for i, category in enumerate(enc.categories_[0]):
    print(f"{category}: {i}")

X_train, X_test, expr_train, expr_test, y_oht_train, y_oht_test, y_train, y_test = train_test_split(X, expr, y_oht, y, test_size=0.3,
                                                                                                    random_state=2024, stratify=y)
X_test, X_val, expr_test, expr_val, y_oht_test, y_oht_val, y_test, y_val = train_test_split(X_test, expr_test, y_oht_test, y_test, 
                                                                                                test_size=0.5, random_state=2024, 
                                                                                            stratify=y_test)

use_cuda = True
device = torch.device("cuda:3" if use_cuda else "cpu")

X_train = torch.from_numpy(X_train).to(device)
X_val = torch.from_numpy(X_val).to(device)
X_test = torch.from_numpy(X_test).to(device)
expr_train = torch.from_numpy(expr_train).to(device)
expr_val = torch.from_numpy(expr_val).to(device)
expr_test = torch.from_numpy(expr_test).to(device)
y_train = torch.from_numpy(y_oht_train).to(device)
y_val = torch.from_numpy(y_oht_val).to(device)
y_test = torch.from_numpy(y_oht_test).to(device)

edge = pd.read_csv('./predict_status/string_filtered.txt', sep='\t')
edge = torch.cat((torch.from_numpy(np.array(edge)).t(), torch.from_numpy(np.array(edge)).t().flip([0])), dim=1).to(device)

# Def

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
    def __init__(self, num_genes, num_peaks, num_tf, cut):
        super(Net, self).__init__()
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
        #self.fc1 = nn.Linear(self.num_genes, 100)
        self.conv_activate = nn.ReLU()
        self.out = nn.Linear(self.num_genes*self.gene_dim, 3)
        
        #self.initialize_parameters()
        
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
        #x = F.dropout(x, p=0.5)
        x = self.conv_activate(self.conv(x, edge))
        #x = F.dropout(x, p=0.5)
        x = x.reshape(x.shape[0], -1)
        out = self.out(x)
        return x_cat, x, out

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    #train_loss = 0
    for batch_idx, (data, expr, target) in enumerate(train_loader):
        data, expr, target = data.float(), expr.float(), target.float()
        optimizer.zero_grad()
        expr_hat, cluster_repr, output = model(data)
        loss_out = out_criterion(output, target)
        #loss_pseudo = expr_criterion(output[:, -1], target[:, -1])
        loss_expr = expr_criterion(expr_hat, expr)
        loss = loss_out #+ 0.7*loss_pseudo
        #loss = loss_pseudo
        loss.backward()
        optimizer.step()
        #model.fc1.weight.data = model.fc1.weight.mul(torch.repeat_interleave(mask.to(device), 4, dim=0))
#         if batch_idx % batchsize == 0:
#             print('\nTrain Epoch: {} [{}/{} ({:.0f}%)], Expr loss: {:.6f}, Cluster loss: {:.6f}'.
#                   format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss_expr.item(), loss_out.item()))
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
            data, expr, target = data.float(), expr.float(), target.float()
            expr_hat, cluster_repr, output = model(data)
            
            loss_out = out_criterion(output, target)
            #loss_pseudo = expr_criterion(output[:, -1], target[:, -1])
            loss_expr = expr_criterion(expr_hat, expr)
            loss = loss_out
            #loss = loss_pseudo
            test_loss = test_loss+loss.item()

            target_cluster = target.argmax(dim=1)
            output_cluster = output.softmax(dim=1)
            outputs = torch.cat((outputs, output_cluster), dim=0)
            targets = torch.cat((targets, target_cluster), dim=0)
            #target_pseudo = target[:, -1]
            #output_pseudo = output[:, -1]
            #outputs_pseudo = torch.cat((outputs_pseudo, output_pseudo), dim=0)
            #targets_pseudo = torch.cat((targets_pseudo, target_pseudo), dim=0)
        f1_score = multiclass_f1_score(outputs, targets, num_classes=num_clusters)
        #pearsonr, _ = stats.pearsonr(targets_pseudo.detach().cpu().numpy(), outputs_pseudo.detach().cpu().numpy())

    return(f1_score, test_loss)

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

batchsize = 128

train_dataset = TensorDataset(X_train, expr_train, y_train)
val_dataset = TensorDataset(X_val, expr_val, y_val)
test_dataset = TensorDataset(X_test, expr_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

print('ReLU: ')
for rep in range(10):
    model = Net(num_genes=expr.shape[1], num_peaks=num_peaks, num_tf=X.shape[1], cut=cut).to(device)
    out_criterion = nn.CrossEntropyLoss()
    expr_criterion=nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    train_losses = []
    test_losses = []
    val_losses = []
    es = 0
    best_score = 1000
    for epoch in range(1, 2000+1):
        model.train()
        train(model, device, train_loader, optimizer, epoch)

        model.eval()
        train_score, train_loss = test(model, device, train_loader, num_clusters=y_oht.shape[1])
        val_score, val_loss = test(model, device, val_loader, num_clusters=y_oht.shape[1])
        test_score, test_loss = test(model, device, test_loader, num_clusters=y_oht.shape[1])
#         print('Epoch: ' + str(epoch))
#         print('Train Loss: {:.6f}, Val Loss: {:.6f}, Test Loss: {:.6f}'.format(train_loss, val_loss, test_loss))
#         print('Train F1: {:.6f}, Val F1: {:.6f}, Test F1: {:.6f}'.format(train_score, val_score, test_score))
        train_losses.append(train_loss/X_train.shape[0])
        val_losses.append(val_loss/X_val.shape[0])
        test_losses.append(test_loss/X_test.shape[0])

        if val_loss < best_score:
            best_score = val_loss
            best_val = val_score
            best_test = test_score
            es = 0
            torch.save(model.state_dict(), './predict_status/model.GCN.string.'+str(rep)+'.pt')
        else:
            es += 1
            #print("Counter {} of 50".format(es))

            if es > 50:
                break   
    print("Val: ", best_val.item(), "Test: ", best_test.item())

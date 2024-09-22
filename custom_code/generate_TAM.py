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

def init_with_precomputed_matrix(tensor, precomputed_matrix):
    tensor.data = precomputed_matrix.clone()

class Net(nn.Module):
    def __init__(self, num_peak, num_tf, W):
        super(Net, self).__init__()
        self.num_peak = num_peak
        self.num_tf = num_tf

        self.W = nn.Parameter(torch.Tensor(num_tf, num_peak))
        init_with_precomputed_matrix(self.W, W)

        self.fc1 = nn.Linear(self.num_peak * self.num_tf, 1)
        self.fc1_activate = nn.ReLU()
        
    def forward(self, x):
        x = x * abs(self.W.repeat(x.shape[0], 1, 1))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, default='./', help='Path of the data files (all data files should be saved under a same directory)')
parser.add_argument('--savepath', type=str, default='./', help='Path of the result folder')
parser.add_argument('--ref', type=str, default='hg19', help='The reference genome. Currently, we only support the usage of hg19 and mm10. You may modify the source code and download relevant files if using other reference genomes.')
parser.add_argument('--device', type=str, default='cuda:1', help='cpu or cuda:0, etc.')
args = parser.parse_args()

os.chdir(args.filepath)

### Loading data
print('\nLoading data...')
df_x = pd.read_csv('./atac.csv', index_col=0).transpose()
df_y = pd.read_csv('./rna.csv', index_col=0).transpose()
df_peaks = pd.DataFrame(df_x.columns)[0].str.split('-',expand=True) 
df_peaks = df_peaks.rename(columns={0: "chrom", 1: "start", 2: "end"})
df_peaks["start"] = pd.to_numeric(df_peaks["start"])
df_peaks["end"] = pd.to_numeric(df_peaks["end"])
annoname = args.ref+"_geneanno.txt"
geneanno = pd.read_csv('./'+annoname, sep='\t')
geneanno = geneanno.drop_duplicates(subset=['Gene name'])

if args.ref=="hg19":
    filename = "JASPAR_motifs_pfm_homosapiens"
elif args.ref=="mm10":
    filename = "JASPAR_motifs_pfm_mouse"
else:
    sys.exit("The provided reference genome version not supported. Currently, we only support hg19 and mm10. For other usage, please modify the source code.")

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

df = pd.read_csv(args.savepath+'processing/pearson.testset.txt', sep='\t', header=None, index_col=0)
df = df.dropna(axis=1, how='all')
genes_filtered = df.index.to_list()
print(str(len(genes_filtered))+" genes in total.")
print("Done!")

# Generating TAM
print("\nGenerating TAM...")
X_all = []
cell_by_tf_all = []
cell_by_peak_all = []
W_all = []
num_peaks = []
peaks_all = pd.DataFrame()
count = 0
i=0
for marker in genes_filtered:
    i = i + 1
    print(str(i)+": "+marker)
    length = 250000
    anno = geneanno.loc[geneanno['Gene name']==marker,:]
    chrom = 'chr'+ str(anno['Chromosome/scaffold name'].values[0])
    start = anno['Gene start (bp)'].values[0]
    end = anno['Gene end (bp)'].values[0]
    strand = anno['Strand'].values[0]
    if strand==1:
        peaks = df_peaks.loc[(df_peaks['chrom']==chrom) & ((df_peaks['start']>=start-length) & (df_peaks['start']<start+length) | 
                                                           (df_peaks['end']>start-length) & (df_peaks['end']<=start+length)),:]
    else:
        peaks = df_peaks.loc[(df_peaks['chrom']==chrom) & ((df_peaks['start']>=end-length) & (df_peaks['start']<end+length) | 
                                                           (df_peaks['end']>end-length) & (df_peaks['end']<=end+length)),:]
    cellcount = (df_x.iloc[:, peaks.index]!=0).astype(int).sum(axis=1)
    cells = cellcount[cellcount!=0].index

    cells_ind = np.where(np.array(cellcount)!=0)[0]
    tfs_use = tfs_kept.copy()
    cell_by_tf_mat = np.array(df_y.loc[:, tfs_use], dtype='float32')
    cell_by_tf_all = cell_by_tf_mat
    cell_by_peak_mat = np.array(df_x.iloc[:, peaks.index], dtype='float32')
    order = np.argsort(np.array(peaks.loc[:,'start']))
    cell_by_peak_mat = cell_by_peak_mat[:, order]
    cell_by_peak_mat = cell_by_peak_mat/np.max(cell_by_peak_mat)
    cell_by_peak_all.append(cell_by_peak_mat)
    cell_by_tf_mat_concat = cell_by_tf_mat.reshape(cell_by_tf_mat.shape[0], cell_by_tf_mat.shape[1], 1)
    cell_by_tf_mat = np.repeat(cell_by_tf_mat[:, :, np.newaxis], peaks.shape[0], axis=2)
    cell_by_peak_mat = np.repeat(cell_by_peak_mat[:, np.newaxis, :], len(tfs_use), axis=1)
    X = cell_by_tf_mat * cell_by_peak_mat
    
    W = torch.ones(X.shape[1], X.shape[2])
    model = Net(num_peak=X.shape[2], num_tf=X.shape[1], W=W)
    num = np.argsort(np.array(df.loc[marker]))[::-1][0]
    model.load_state_dict(torch.load(args.savepath+'processing/gene_net/'+marker+'.'+str(num)+'.pt', map_location=args.device))
    W = abs(model.W.data).clone().detach().numpy()[:, order]
    W_all.append(W)
    X = (X*W)#.astype('float16')
    print(X.shape)    
    X_all.append(X)
    peaks_all = peaks_all.append(peaks.iloc[order, :].reset_index(drop=True))
    num_peaks.append(X.shape[2])
print("Done!")

print("\nConcatenating TAM of all genes...")
X = np.concatenate(X_all, axis=2)
print(X.shape)
print("Done!")

print("\nConcatenating cCRE of all genes...")
cell_by_peak = np.concatenate(cell_by_peak_all, axis=1)
print(cell_by_peak.shape)
print("Done!")

print("\nConcatenating W of all genes...")
W = np.concatenate(W_all, axis=1)
print(W.shape)
print("Done!")

cell_by_tf = cell_by_tf_all

s = 0
for i in num_peaks:
    s = s + i
if s!=peaks_all.shape[0]:
    sys.exit("Dimension incorrect. Please check the code.")

peaks_all['chrom'] = peaks_all['chrom'].astype(str)
peaks_all['start'] = peaks_all['start'].astype(str)
peaks_all['end'] = peaks_all['end'].astype(str)

print("\nWriting files...")
h5f = h5py.File(args.savepath+'processing/data.h5', 'w')
h5f.create_dataset('X', data=X)
h5f.create_dataset('num_peaks', data=np.array(num_peaks))
h5f.create_dataset('peaks', data=peaks_all.to_numpy())
h5f.create_dataset('cell_by_tf', data=cell_by_tf)
h5f.create_dataset('cell_by_peak', data=cell_by_peak)
h5f.create_dataset('W', data=W)
h5f.close()
print("Done!")
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

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.flatten(), target)
        loss.backward()
        optimizer.step()

                  
def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        output_all = []
        target_all = []
        for data, target in test_loader:
            #data, target = data.to(device), target.to(device)
            target_all.append(target)
            output = model(data)
            output_all.append(output.flatten())
        output_all = torch.cat(output_all, dim=0)
        target_all = torch.cat(target_all, dim=0)
        criterion = nn.MSELoss()
        score = criterion(output_all, target_all).item()
    return(score)

def correlation_score(y_true, y_pred):
    #print(np.corrcoef(y_true, y_pred))
    return np.corrcoef(y_true, y_pred)[1, 0]

def spearman_correlation(y_true, y_pred):
    statistic, pvalue = stats.spearmanr(y_true, y_pred)
    return abs(statistic)

def pearson_correlation(y_true, y_pred):
    statistic, pvalue = stats.pearsonr(y_true, y_pred)
    return abs(statistic[0])

    
    
parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, default='./', help='Path of the data files (all data files should be saved under a same directory)')
parser.add_argument('--savepath', type=str, default='./', help='Path of the result folder')
parser.add_argument('--ref', type=str, default='hg19', help='The reference genome. Currently, we only support the usage of hg19 and mm10. You may modify the source code and download relevant files if using other reference genomes.')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:0, etc.')
parser.add_argument('--batchsize', type=int, default=256, help='Batchsize when learning TF-cCRE interactions')
parser.add_argument('--lr', type=int, default=0.001, help='Learning rate when learning TF-cCRE interactions')

args = parser.parse_args()


### Processing PFM matrices
print("Processing PFM matrices...")
os.chdir(args.filepath)
if args.ref=="hg19":
    filename = "JASPAR_motifs_pfm_homosapiens"
elif args.ref=="mm10":
    filename = "JASPAR_motifs_pfm_mouse"
else:
    sys.exit("The provided reference genome version not supported. Currently, we only support hg19 and mm10. For other usage, please modify the source code.")

motif_files = [f for f in os.listdir('./' + filename) if f.endswith('.transfac')]

if not os.path.exists('./'+filename+'/pfm.np/'):
    os.makedirs('./'+filename+'/pfm.np/')

for i in range(len(motif_files)):
    #print('./'+filename+'/'+motif_files[i])
    motif_file = open('./'+filename+'/'+motif_files[i])
    flag = False
    mat = []
    for lines in motif_file:
        line = lines.split()
        if 'ID' in line:
            motif = line[1]
        if '01' in line:
            flag = True
        if 'XX' in line:
            flag = False
        if flag==True:
            mat.append(line[1:])
    mat = np.array(mat, dtype = 'float')
    np.save("./"+filename+"/pfm.np/"+motif+".npy", mat)
print("Done!")

### Learning W in the TAM matrices
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
markers = pd.read_csv("./genes.txt", sep='\t', header=None).iloc[:,0].drop_duplicates().to_list()
print("Done!")

print("\nTraining/validation/test set splitting...")
y_label = pd.read_csv('./label.csv')
y_label = y_label.iloc[:, 0].to_list()
sample_num = [i for i in range(len(y_label))]
sample_train, sample_test, y_train, y_test = train_test_split(sample_num, y_label, test_size=0.3, random_state=2024, stratify=y_label)
sample_test, sample_val, y_test, y_val = train_test_split(sample_test, y_test, test_size=0.5, random_state=2024, stratify=y_test)
print("Done!")

print("\nLearning TF-cCRE interactions for each gene...")
print("This step takes a relatively long time (about 2 mins per gene on gpu). Please be patient...")
motif_files = os.listdir('./'+filename+'/pfm.np')
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
print(str(len(tfs_kept))+' TFs included')


# Run
if not os.path.exists(args.savepath+'processing/gene_net/'):
    os.makedirs(args.savepath+'processing/gene_net/')
if os.path.exists(args.savepath+'processing/pearson.testset.txt'):
    #os.remove(args.savepath+'processing/pearson.testset.txt')
    df_exist = pd.read_csv(args.savepath+'processing/pearson.testset.txt', sep='\t', header=None, index_col=0)
    genes_exist = df_exist.index.to_list()
    print('Filtering genes already been tested...')
    markers = [i for i in markers if i not in genes_exist]
    print(str(len(markers))+' genes left.')
    
test_genes = []
num = 0
print('Filtering genes not included in the data...')
markers_1 = [i for i in markers if i in df_y.columns]
print(str(len(markers)-len(markers_1))+' genes filtered.')
print('Filtering genes that are TFs...')
markers = [i for i in markers_1 if i not in tfs_kept]
print(str(len(markers_1)-len(markers))+' genes filtered.')
print(str(len(markers))+' genes left.')
for marker in markers:
    num = num + 1
    print(str(num)+": "+marker)
    length = 250000
    anno = geneanno.loc[geneanno['Gene name']==marker,:]
    if anno.shape[0]>0:
        test_genes.append(marker)
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
        #print(len(cells))
        if len(cells)<0.1*df_x.shape[0]:
            print('Too few cells.')
            print('\n')
            continue
    else:
        print('No such gene in the gene annotation file.')
        print('\n')
        continue
    
    cells_ind = np.where(np.array(cellcount)!=0)[0]
    y = np.array(df_y.loc[:, marker], dtype='float32').flatten()

    tfs_use = tfs_kept.copy()
    cell_by_tf_mat = np.array(df_y.loc[:, tfs_use], dtype='float32')
    cell_by_peak_mat = np.array(df_x.iloc[:, peaks.index], dtype='float32')

    # Scale data
    cell_by_peak_mat = cell_by_peak_mat/np.max(cell_by_peak_mat)

    cell_by_tf_mat_concat = cell_by_tf_mat.reshape(cell_by_tf_mat.shape[0], cell_by_tf_mat.shape[1], 1)
    cell_by_tf_mat = np.repeat(cell_by_tf_mat[:, :, np.newaxis], peaks.shape[0], axis=2)
    cell_by_peak_mat = np.repeat(cell_by_peak_mat[:, np.newaxis, :], len(tfs_use), axis=1)
    X = cell_by_tf_mat * cell_by_peak_mat

    y_std = (y - y[cells_ind].min()) / (y[cells_ind].max() - y[cells_ind].min())
    y = y_std * (1 - (-1)) + (-1)
        
    train_ind = [i for i in sample_train if i in cells_ind]
    test_ind = [i for i in sample_test if i in cells_ind]
    val_ind = [i for i in sample_val if i in cells_ind]
    X_train = X[train_ind, :, :]
    X_test = X[test_ind, :, :]
    X_val = X[val_ind, :, :]
    y_train = y[train_ind]
    y_test = y[test_ind]
    y_val = y[val_ind]
           
    if sum(y_train==-1)==len(y_train) or sum(y_test==-1)==len(y_test) or sum(y_val==-1)==len(y_val):
        print('Too many zeros.')
        print('\n')
        continue

    device = torch.device(args.device)
    batch_size = args.batchsize

    X_train = torch.from_numpy(X_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    
    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    pearsons = []
    for time in range(2):
        print("Train "+str(time)+"...")
        W = torch.ones(X.shape[1], X.shape[2]).to(device)
        model = Net(num_peak=X.shape[2], num_tf=X.shape[1], W=W).to(device)
        criterion=nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        train_loss = []
        test_loss = []
        val_loss = []
        best_score = 100000
        for epoch in range(1, 3000 + 1):
            model.train()
            train(model, device, train_loader, optimizer, epoch)
            train_score = test(model, device, train_loader)
            val_score = test(model, device, val_loader)
            test_score = test(model, device, test_loader)

            model.eval()
            train_loss.append(criterion(model(X_train).flatten(), y_train).detach().cpu().numpy().tolist())
            test_loss.append(criterion(model(X_test).flatten(), y_test).detach().cpu().numpy().tolist())
            val_loss.append(criterion(model(X_val).flatten(), y_val).detach().cpu().numpy().tolist())

            if val_score < best_score:
                best_score = val_score
                es = 0
                torch.save(model.state_dict(), args.savepath+'processing/gene_net/'+marker+'.'+str(time)+'.pt')
            else:
                es += 1
                if es > 300:
                    break
        model.load_state_dict(torch.load(args.savepath+'processing/gene_net/'+marker+'.'+str(time)+'.pt'))
        output = model(X_test)
        final_score = pearson_correlation(y_test.detach().cpu().numpy().flatten(), output.detach().cpu().numpy())
        print("Test set PCC: "+str(final_score))
        model.load_state_dict(torch.load(args.savepath+'processing/gene_net/'+marker+'.'+str(time)+'.pt'))
        output = model(X_test)
        pearsons.append(final_score)
        
    with open(args.savepath+'processing/pearson.testset.txt',"a") as file:
        file.write(marker+'\t')
        for pearson in pearsons:
            file.write(str(pearson)+'\t')
        file.write('\n')
    print('\n')
        
print("Done!")


# This script describes the first step of training to learn the TF-cCRE interaction matrix W.
# Before running the following codes, prepare or download the following files:
# 1. hg19 reference genome "hg19.fa" from the UCSC genome browser (https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz)
# 2. Processed genomic annotation file "hg19_geneanno.txt" from the "data" folder in this github repository
# 3. PFM mtrices of motifs obtained from "0.process_pfm_from_JASPAR.py"
# 4. Aggregated RNA and ATAC data, DE genes obtained from "0.preprocess_T2D.R"

import scanpy as sc
import pandas as pd
import numpy as np
from verstack import stratified_continuous_split
from pyfaidx import Fasta
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
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import h5py
import seaborn as sns
import os
os.chdir('/nfs/public/xixi/scRegulate/T2D')

# Data preprocess
df_x = pd.read_csv('./data/beta.atac.aggregate_30cells.csv', index_col=0).transpose()
df_y = pd.read_csv('./data/beta.rna.aggregate_30cells.csv', index_col=0).transpose()
df_peaks = pd.DataFrame(df_x.columns)[0].str.split('-',expand=True) 
df_peaks = df_peaks.rename(columns={0: "chrom", 1: "start", 2: "end"})
df_peaks["chrom"] = 'chr' + df_peaks["chrom"]
df_peaks["start"] = pd.to_numeric(df_peaks["start"])
df_peaks["end"] = pd.to_numeric(df_peaks["end"])
geneanno = pd.read_csv('../../ref_genome/hg19_geneanno.txt', sep='\t')
geneanno = geneanno.drop_duplicates(subset=['Gene name'])
markers = pd.read_csv("./data/markers_status.txt", sep='\t')
markers = markers['gene'].drop_duplicates().to_list()

# Prepare data
motif_files = os.listdir('../../ref_genome/JASPAR_motifs_pfm_homosapiens/pfm.np')

tfs_kept = []
for i in list(motif_files):
    tf = i.split('.')[-2]
    if tf in df_y.columns:
        if tf in tfs_kept:
            continue
            
        tfs_kept.append(tf)
tfs_kept = sorted(tfs_kept)

# Def
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
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #print(output.shape)
        #print(target.shape)
        loss = criterion(output.flatten(), target)
        loss.backward()
        optimizer.step()

                  
def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        output_all = []
        target_all = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
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


# Run
test_genes = []
test_pearsons = []
for marker in markers:
    length = 250000
    anno = geneanno.loc[geneanno['Gene name']==marker,:]
    if anno.shape[0]>0:
        test_genes.append(marker)
        print(marker)
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
        if len(cells)<100:
            print('Too few cells.')
            continue
    else:
        print(marker)
        print('No such gene in the gene annotation file.')
        continue
    
    cells_ind = np.where(np.array(cellcount)!=0)[0]
    y = np.array(df_y.loc[:, marker], dtype='float32')
    y = y[cells_ind].flatten()

    tfs_use = tfs_kept.copy()
    if marker in tfs_kept:
        tfs_use.remove(marker)
    cell_by_tf_mat = df_y.loc[:, tfs_use]
    cell_by_tf_mat = np.array(cell_by_tf_mat.iloc[cells_ind, :], dtype='float32')
    #cell_by_peak_mat = (np.array(df_x.iloc[:, peaks.index], dtype='float32')!=0).astype(int)
    cell_by_peak_mat = np.array(df_x.iloc[:, peaks.index], dtype='float32')
    cell_by_peak_mat = cell_by_peak_mat[cells_ind, :]

    # Scale data
    cell_by_peak_mat = cell_by_peak_mat/np.max(cell_by_peak_mat)

    cell_by_tf_mat_concat = cell_by_tf_mat.reshape(cell_by_tf_mat.shape[0], cell_by_tf_mat.shape[1], 1)
    cell_by_tf_mat = np.repeat(cell_by_tf_mat[:, :, np.newaxis], peaks.shape[0], axis=2)
    cell_by_peak_mat = np.repeat(cell_by_peak_mat[:, np.newaxis, :], len(tfs_use), axis=1)
    X = cell_by_tf_mat * cell_by_peak_mat

    y_std = (y - y.min()) / (y.max() - y.min())
    y = y_std * (1 - (-1)) + (-1)
    
    if sum(y==-1)>500:
        print('Too few cells with target gene expression.')
        continue

    bins = stratified_continuous_split.estimate_nbins(y)
    y_binned = np.digitize(y, bins)
    y_binned = stratified_continuous_split.combine_single_valued_bins(y_binned)
    try:
        X_train, X_test, y_train, y_test, ybin_train, ybin_test = train_test_split(X, y, y_binned, test_size=0.2, random_state=2023, stratify=y_binned)
    except:
        print("Train-test split not stratified.")
        X_train, X_test, y_train, y_test, ybin_train, ybin_test = train_test_split(X, y, y_binned, test_size=0.2, random_state=2023)
    try:
        X_train, X_val, y_train, y_val, ybin_train, ybin_val = train_test_split(X_train, y_train, ybin_train, test_size=0.2, random_state=2023, stratify=ybin_train)
    except:
        print("Train-val split not stratified.")
        X_train, X_val, y_train, y_val, ybin_train, ybin_val = train_test_split(X_train, y_train, ybin_train, test_size=0.2, random_state=2023)

    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_val = torch.from_numpy(y_val).float()
    y_test = torch.from_numpy(y_test).float()
    
    use_cuda = True
    device = torch.device("cuda:1" if use_cuda else "cpu")
    batch_size = 256

    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    pearsons = []
    for time in range(3):
        print("Train "+str(time)+"...")
        W = torch.ones(X.shape[1], X.shape[2]).to(device)
        model = Net(num_peak=X.shape[2], num_tf=X.shape[1], W=W).to(device)
        #model.W.requires_grad=False
        criterion=nn.MSELoss()
        #criterion=nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        train_loss = []
        test_loss = []
        val_loss = []
        #best_score = 0
        best_score = 100000
        for epoch in range(1, 3000 + 1):
            model.train()
            train(model, device, train_loader, optimizer, epoch)
            train_score = test(model, device, train_loader)
            val_score = test(model, device, val_loader)
            test_score = test(model, device, test_loader)
            #print('Train correlation: {:.6f}, Val correlation: {:.6f}, Test correlation: {:.6f}'.format(train_score, val_score, test_score))
            #print('Train RMSE: {:.6f}, Val RMSE: {:.6f}, Test RMSE: {:.6f}'.format(train_score, val_score, test_score))

            model.eval()
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            train_loss.append(criterion(model(X_train).flatten(), y_train).detach().cpu().numpy().tolist())
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            test_loss.append(criterion(model(X_test).flatten(), y_test).detach().cpu().numpy().tolist())
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            val_loss.append(criterion(model(X_val).flatten(), y_val).detach().cpu().numpy().tolist())

            #if val_score > best_score:
            if val_score < best_score:
                best_score = val_score
                es = 0
                torch.save(model.state_dict(), './nn.best.feature6.learnW/'+marker+'.'+str(time)+'.pt')
            else:
                es += 1
                #print("Counter {} of 100".format(es))

                if es > 300:
                    print("Early stopping with best_score: ", best_score, "and val_score for this epoch: ", val_score, "...")
                    break
        model.load_state_dict(torch.load('./nn.best.feature6.learnW/'+marker+'.'+str(time)+'.pt'))
        X_test = X_test.to(device)
        output = model(X_test)
        pearsons.append(pearson_correlation(y_test.detach().cpu().numpy().flatten(), output.detach().cpu().numpy()))
        
    with open("./pearson.testset.feature6_learnW.txt","a") as file:
        file.write(marker+'\t')
        for pearson in pearsons:
            file.write(str(pearson)+'\t')
        file.write('\n')
        
    test_pearsons.append(pearsons)
print("All Done!")
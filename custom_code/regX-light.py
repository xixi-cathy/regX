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
import scipy.stats as ss
import warnings
warnings.filterwarnings('ignore')


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

        self.fc1 = nn.Linear(self.num_peak+self.num_tf, 1)
        self.fc1_activate = nn.ReLU()
        self.abs = nn.ReLU()
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

class Net_GCN(nn.Module):
    def __init__(self, num_genes, num_peaks, num_tf, cut, num_output):
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
            x_sub = x[:, self.cut[i]:self.cut[i+1]]
            x_sub = self.subnet_modules[i](x_sub)
            x_cat = torch.cat((x_cat, x_sub), dim=1)

        x_cat = self.cat_activate(x_cat)
        x = torch.unsqueeze(x_cat, 2)
        x = self.conv_activate(self.conv(x, edge))
        x = x.reshape(x.shape[0], -1)
        out = self.out(x)
        return out

class Net_SAGEConv(nn.Module):
    def __init__(self, num_genes, num_peaks, num_tf, cut, num_output):
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
        nn.init.constant_(weight_l[1][0], 2*torch.rand(1)[0]+0.1)
        #nn.init.uniform_(weight_l[1][0], -2, 2)
        weight_r = self.conv.lin_r.weight
        nn.init.constant_(weight_r[0][0], 2*torch.rand(1)[0]+0.1)
        nn.init.constant_(weight_r[1][0], 2*torch.rand(1)[0]+0.1)
        #nn.init.uniform_(weight_r[1][0], -2, 2)

    def forward(self, x):
        x_cat = torch.zeros(x.shape[0], 0).to(device)
        for i in range(len(self.subnet_modules)):
            x_sub = x[:, self.cut[i]:self.cut[i+1]]
            x_sub = self.subnet_modules[i](x_sub)
            x_cat = torch.cat((x_cat, x_sub), dim=1)

        x_cat = self.cat_activate(x_cat)
        x = torch.unsqueeze(x_cat, 2)
        x = self.conv_activate(self.conv(x, edge))
        x = x.reshape(x.shape[0], -1)
        out = self.out(x)
        return out    
    
    
class Net_GAT(nn.Module):
    def __init__(self, genes, num_peaks, num_tf, cut, mask, num_gos, num_output):
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
            x_sub = x[:, self.cut[i]:self.cut[i+1]]
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

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float(), target.float()
        optimizer.zero_grad()
        output = model(data)
        loss = out_criterion(output, target)
        loss.backward()
        optimizer.step()

                  
def test(model, device, test_loader, num_clusters):
    model.eval()
    with torch.no_grad():
        outputs = torch.zeros(0, num_clusters).to(device)
        targets = torch.zeros(0).to(device)
        test_loss = 0
        for data, target in test_loader:
            data, target = data.float(), target.float()
            output = model(data)            
            loss = out_criterion(output, target)
            test_loss = test_loss+loss.item()
            target_cluster = target.argmax(dim=1)
            output_cluster = output.softmax(dim=1)
            outputs = torch.cat((outputs, output_cluster), dim=0)
            targets = torch.cat((targets, target_cluster), dim=0)
        f1_score = multiclass_f1_score(outputs, targets, num_classes=num_clusters)
    return(f1_score, test_loss)    
    

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, default='./', help='Path of the data files (all data files should be saved under a same directory)')
parser.add_argument('--savepath', type=str, default='./', help='Path of the result folder')
parser.add_argument('--ref', type=str, default='hg19', help='The reference genome. Currently, we only support hg19 and mm10. You may modify the source code and download relevant files if using other reference genomes.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for training. cpu or cuda:2, etc.')
parser.add_argument('--model', type=str, default='GCN', help='GNN type. GCN, GAT or SAGEConv')
parser.add_argument('--replicates', type=int, default=5, help='Number of replicates for training. The final TF/cCRE prioritization results will take the average of these replicates. We suggest 5-10 replicates.')
parser.add_argument('--batchsize', type=int, default=128, help='Batchsize when training regX')
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
else:
    print("PFM matrices already processed.")

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

markers = pd.read_csv("./genes.txt", sep='\t', header=None).iloc[:,0].drop_duplicates().to_list()
print('Filtering genes not included in the data...')
markers_1 = [i for i in markers if i in df_y.columns]
print(str(len(markers)-len(markers_1))+' genes filtered.')
print('Filtering genes that are TFs...')
markers = [i for i in markers_1 if i not in tfs_kept]
print(str(len(markers_1)-len(markers))+' genes filtered.')
print(str(len(markers))+' genes left.')
print("Done!")

### Generating input features
print("\nGenerating training data...")
if os.path.exists(args.savepath+'processing/data_concat.h5'):
    print("Input feature file already exists.")
else:
    genes_filtered = []
    X_all = []
    cell_by_tf_all = []
    cell_by_peak_all = []
    num_peaks = []
    peaks_all = pd.DataFrame()
    count = 0
    i=0
    for marker in markers:
        i = i + 1
        #print(str(i)+": "+marker)
        length = 250000
        anno = geneanno.loc[geneanno['Gene name']==marker,:]
        if anno.shape[0]>0:
            genes_filtered.append(marker)
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
            X = np.hstack([cell_by_tf_mat, cell_by_peak_mat])
            #print(X.shape)

            X_all.append(X)
            peaks_all = peaks_all.append(peaks.iloc[order, :].reset_index(drop=True))
            num_peaks.append(cell_by_peak_mat.shape[1])
            cell_by_peak_all.append(cell_by_peak_mat)
        else:
            #print('No such gene in the gene annotation file.')
            continue

    print("Concatenating TAM of all genes...")
    X = np.concatenate(X_all, axis=1)
    print(X.shape)

    print("Concatenating cCRE of all genes...")
    cell_by_peak = np.concatenate(cell_by_peak_all, axis=1)
    print(cell_by_peak.shape)

    cell_by_tf = cell_by_tf_all

    s = 0
    for i in num_peaks:
        s = s + i
    if s!=peaks_all.shape[0]:
        sys.exit("Dimension incorrect. Please check the code.")

    peaks_all['chrom'] = peaks_all['chrom'].astype(str)
    peaks_all['start'] = peaks_all['start'].astype(str)
    peaks_all['end'] = peaks_all['end'].astype(str)
    print("Done!")

    print("\nWriting files...")
    h5f = h5py.File(args.savepath+'processing/data_concat.h5', 'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('num_peaks', data=np.array(num_peaks))
    h5f.create_dataset('peaks', data=peaks_all.to_numpy())
    h5f.create_dataset('cell_by_tf', data=cell_by_tf)
    h5f.create_dataset('cell_by_peak', data=cell_by_peak)
    h5f.close()
    pd.DataFrame(genes_filtered).to_csv(args.savepath+'processing/concat.genes_filtered.txt', index=False, header=False)
    print("Done!")

print('\nLoading training data...')
h5f = h5py.File(args.savepath+'processing/data_concat.h5', 'r')
X = h5f['X'][:]
num_peaks = h5f['num_peaks'][:]
cell_by_tf = h5f['cell_by_tf'][:]
cell_by_peak = h5f['cell_by_peak'][:]
h5f.close()

genes_filtered = pd.read_csv(args.savepath+'processing/concat.genes_filtered.txt', header=None)[0].to_list()

cut = [0]
s = 0
for i in num_peaks:
    s = s+i+len(tfs_kept)
    cut.append(s)
print("Done!")

print("\nOriginal labels and corresponding channels:")
y = pd.read_csv('./label.csv')
y = np.array(y.iloc[:, 0].to_list())
y = y.reshape((len(y), 1))
enc = OneHotEncoder(handle_unknown='ignore')
y_oht = enc.fit_transform(y).toarray()
for i, category in enumerate(enc.categories_[0]):
    print(f"{category}: {i}")

print("\nTraining/validation/test set splitting...")
X_train, X_test, y_oht_train, y_oht_test, y_train, y_test = train_test_split(X, y_oht, y, test_size=0.3,random_state=2024, stratify=y)
X_test, X_val, y_oht_test, y_oht_val, y_test, y_val = train_test_split(X_test, y_oht_test, y_test, test_size=0.5, random_state=2024, stratify=y_test)
indices_train, indices_test, y_label_train, y_label_test= train_test_split(np.arange(X.shape[0]), y, test_size=0.3, random_state=2024, stratify=y)
indices_test, indices_val, y_label_test, y_label_val= train_test_split(indices_test, y_label_test, test_size=0.5, random_state=2024, stratify=y_label_test)
print("Done!")

device = torch.device(args.device)

X_train = torch.from_numpy(X_train).to(device)
X_val = torch.from_numpy(X_val).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_train = torch.from_numpy(y_oht_train).to(device)
y_val = torch.from_numpy(y_oht_val).to(device)
y_test = torch.from_numpy(y_oht_test).to(device)

print("\nProcessing network edges...")
if args.model=='GCN' or args.model=='SAGEConv':
    if not os.path.exists(args.savepath+'processing/ppi_filtered_concat.txt'):
        print("Processing STRING PPI...")
        filepath = args.filepath
        savepath = args.savepath
        if args.ref=='hg19':
            species = 'human'
            ppi = pd.read_csv(os.path.join(filepath, '9606.protein.links.v12.0.txt'), sep=' ')
            info = pd.read_csv(os.path.join(filepath, '9606.protein.info.v12.0.txt'), sep='\t')
        elif args.ref=='mm10':
            species = 'mouse'
            ppi = pd.read_csv(os.path.join(filepath, '10090.protein.links.v12.0.txt'), sep=' ')
            info = pd.read_csv(os.path.join(filepath, '10090.protein.info.v12.0.txt'), sep='\t')

        genenames = pd.Series(info['preferred_name'].values, index=info['#string_protein_id']).to_dict()
        ppi['protein1'] = ppi['protein1'].map(genenames)
        ppi['protein2'] = ppi['protein2'].map(genenames)

        genes = genes_filtered
        print(f"{len(genes)} genes in total")
        inds = pd.Series(range(len(genes)), index=genes).to_dict()

        edges = ppi[(ppi['protein1'].isin(genes)) & (ppi['protein2'].isin(genes))][['protein1', 'protein2']].drop_duplicates()
        edges['protein1'] = edges['protein1'].map(inds)
        edges['protein2'] = edges['protein2'].map(inds)
        edges = edges.drop_duplicates()
        edges.to_csv(os.path.join(savepath, 'processing/ppi_filtered_concat.txt'), sep='\t', index=False)
    edge = pd.read_csv(args.savepath+'processing/ppi_filtered_concat.txt', sep='\t')
    edge = torch.cat((torch.from_numpy(np.array(edge)).t(), torch.from_numpy(np.array(edge)).t().flip([0])), dim=1).to(device)
    print("Done!")
elif args.model=='GAT':
    go = pd.read_csv('./go_filtered.txt', sep='\t')
    goa = pd.read_csv('./goa_filtered.txt', sep='\t')

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
    
batchsize = args.batchsize

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

print('\nTraining started...')
if args.model=="GCN":
    print("Model: GCN, batchsize: "+str(args.batchsize)+", lr: "+str(args.lr))
if args.model=="SAGEConv":
    print("Model: SAGEConv, batchsize: "+str(args.batchsize)+", lr: "+str(args.lr))
if args.model=="GAT":
    print("Model: GAT, batchsize: "+str(args.batchsize)+", lr: "+str(args.lr))
if not os.path.exists(args.savepath+'results/model_'+args.model+'/'):
    os.makedirs(args.savepath+'results/model_'+args.model+'/')

if os.path.exists(args.savepath+'results/model_'+args.model+'/'+'concat.model_'+str(args.batchsize)+'_'+str(args.lr)+'_rep'+str(args.replicates)+'.pt'):
    print(str(args.replicates)+' trained models were found.')
else:    
    testscores = []
    print(str(args.replicates)+" replicates in total")
    for rep in range(args.replicates):
        print("replicate "+str(rep+1)+":")
        if args.model=="GCN":
            model = Net_GCN(num_genes=len(num_peaks), num_peaks=num_peaks, num_tf=len(tfs_kept), cut=cut, num_output=y_oht.shape[1]).to(device)
        if args.model=="SAGEConv":
            model = Net_SAGEConv(num_genes=len(num_peaks), num_peaks=num_peaks, num_tf=len(tfs_kept), cut=cut, num_output=y_oht.shape[1]).to(device)
        if args.model=="GAT":
            model = Net_GAT(genes=genes_filtered, num_peaks=num_peaks, num_tf=len(tfs_kept), cut=cut, mask=mask, num_gos=W_true.shape[0], num_output=y_oht.shape[1]).to(device)

        out_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

        train_losses = []
        test_losses = []
        val_losses = []
        es = 0
        best_score = 1000
        for epoch in range(1, 1500+1):
            model.train()
            train(model, device, train_loader, optimizer, epoch)

            model.eval()
            train_score, train_loss = test(model, device, train_loader, num_clusters=y_oht.shape[1])
            val_score, val_loss = test(model, device, val_loader, num_clusters=y_oht.shape[1])
            test_score, test_loss = test(model, device, test_loader, num_clusters=y_oht.shape[1])
            if epoch%10==0:
                print('Epoch: ' + str(epoch))
                print('Train Loss: {:.6f}, Val Loss: {:.6f}, Test Loss: {:.6f}'.format(train_loss, val_loss, test_loss))
                print('Train F1: {:.6f}, Val F1: {:.6f}, Test F1: {:.6f}'.format(train_score, val_score, test_score))
            train_losses.append(train_loss/X_train.shape[0])
            val_losses.append(val_loss/X_val.shape[0])
            test_losses.append(test_loss/X_test.shape[0])

            if val_loss < best_score:
                best_score = val_loss
                best_val = val_score
                best_test = test_score
                es = 0
                torch.save(model.state_dict(), args.savepath+'results/model_'+args.model+'/'+'concat.model_'+str(args.batchsize)+'_'+str(args.lr)+'_rep'+str(rep+1)+'.pt')
            else:
                es += 1
                #print("Counter {} of 50".format(es))
                if es > 30:
                    break   
        print("Test set F1: ", best_test.item())
        testscores.append(best_test.item())

    with open(args.savepath+'results/concat.f1.testset.txt',"a") as file:
        file.write(args.model+'_'+str(args.batchsize)+'_'+str(args.lr)+'\t')
        for testscore in testscores:
            file.write(str(testscore)+'\t')
        file.write('\n')
    print("Done!")


### Prioritizing driver TFs
print("\nPrioritizing driver TFs...")
# Upregulation
if os.path.exists(args.savepath+'processing/concat.'+args.model+'_TF_up.npy'):
    print("in silico upregulation of TFs have been performed.")
    model_files = [f for f in os.listdir(args.savepath + '/results/model_'+args.model+'/') if f.startswith('concat.model_'+str(args.batchsize)+'_'+str(args.lr))]
    repeats = len(model_files)
    results = np.load(args.savepath+'processing/concat.'+args.model+'_TF_up.npy')
else:
    print("Performing in silico upregulation of TFs...")
    model_files = [f for f in os.listdir(args.savepath + '/results/model_'+args.model+'/') if f.startswith('concat.model_'+str(args.batchsize)+'_'+str(args.lr))]
    repeats = len(model_files)
    print(str(repeats)+" replicates in total")
    num_clusters = y_oht.shape[1]
    results = np.zeros([repeats, X_test.shape[0], len(tfs_kept), num_clusters])
    results_orig = np.zeros([repeats, X_test.shape[0], len(tfs_kept), num_clusters])
    results_perturb = np.zeros([repeats, X_test.shape[0], len(tfs_kept), num_clusters])
    for i in range(repeats):
        print('Calculating replicate '+str(i+1)+":")
        if args.model=="GCN":
            model = Net_GCN(num_genes=len(num_peaks), num_peaks=num_peaks, num_tf=len(tfs_kept), cut=cut, num_output=y_oht.shape[1]).to(device)
        if args.model=="SAGEConv":
            model = Net_SAGEConv(num_genes=len(num_peaks), num_peaks=num_peaks, num_tf=len(tfs_kept), cut=cut, num_output=y_oht.shape[1]).to(device)
        if args.model=="GAT":
            model = Net_GAT(genes=genes_filtered, num_peaks=num_peaks, num_tf=len(tfs_kept), cut=cut, mask=mask, num_gos=W_true.shape[0], num_output=y_oht.shape[1]).to(device)
        model.eval()
        model.load_state_dict(torch.load(args.savepath + '/results/model_'+args.model+'/concat.model_'+str(args.batchsize)+'_'+str(args.lr)+'_rep'+str(i+1)+'.pt', 
                                        map_location=device))
        np.random.seed(123)  

        orig = model(X_test)
        orig = F.softmax(orig)
        for j in range(len(tfs_kept)):
            X_new = torch.clone(X_test).detach()
            for num_gene in range(len(genes_filtered)):
                X_new[:, cut[num_gene]+j] = X_new[:, cut[num_gene]+j]+np.quantile(cell_by_tf[:, j], 0.9)

            perturb = model(X_new)
            perturb = F.softmax(perturb)
            results[i, :, j, :] = (perturb-orig).detach().cpu().numpy()
            results_orig[i, :, j, :] = orig.clone().detach().cpu().numpy()
            results_perturb[i, :, j, :] = perturb.detach().cpu().numpy()
        print("Done!")

    np.save(args.savepath+'processing/concat.'+args.model+'_TF_up.npy', results)
    np.save(args.savepath+'processing/concat.'+args.model+'_TF_up_orig.npy', results_orig)
    np.save(args.savepath+'processing/concat.'+args.model+'_TF_up_perturb.npy', results_perturb)


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
df_results.to_csv(args.savepath+'results/concat.'+args.model+'.TF_rank_upregulation.csv', index=False)
print("Done!")


# Downregulation
if os.path.exists(args.savepath+'processing/concat.'+args.model+'_TF_down.npy'):
    print("in silico downregulation of TFs have been performed.")
    model_files = [f for f in os.listdir(args.savepath + '/results/model_'+args.model+'/') if f.startswith('concat.model_'+str(args.batchsize)+'_'+str(args.lr))]
    repeats = len(model_files)
    results = np.load(args.savepath+'processing/concat.'+args.model+'_TF_down.npy')
else:
    print("Performing in silico downregulation of TFs...")
    model_files = [f for f in os.listdir(args.savepath + '/results/model_'+args.model+'/') if f.startswith('concat.model_'+str(args.batchsize)+'_'+str(args.lr))]
    repeats = len(model_files)
    print(str(repeats)+" replicates in total")
    num_clusters = y_oht.shape[1]
    results = np.zeros([repeats, X_test.shape[0], len(tfs_kept), num_clusters])
    results_orig = np.zeros([repeats, X_test.shape[0], len(tfs_kept), num_clusters])
    results_perturb = np.zeros([repeats, X_test.shape[0], len(tfs_kept), num_clusters])
    for i in range(repeats):
        print('Calculating replicate '+str(i+1)+":")
        if args.model=="GCN":
            model = Net_GCN(num_genes=len(num_peaks), num_peaks=num_peaks, num_tf=len(tfs_kept), cut=cut, num_output=y_oht.shape[1]).to(device)
        if args.model=="SAGEConv":
            model = Net_SAGEConv(num_genes=len(num_peaks), num_peaks=num_peaks, num_tf=len(tfs_kept), cut=cut, num_output=y_oht.shape[1]).to(device)
        if args.model=="GAT":
            model = Net_GAT(genes=genes_filtered, num_peaks=num_peaks, num_tf=len(tfs_kept), cut=cut, mask=mask, num_gos=W_true.shape[0], num_output=y_oht.shape[1]).to(device)
        model.eval()
        model.load_state_dict(torch.load(args.savepath + '/results/model_'+args.model+'/concat.model_'+str(args.batchsize)+'_'+str(args.lr)+'_rep'+str(i+1)+'.pt', 
                                        map_location=device))
        np.random.seed(123)  

        orig = model(X_test)
        orig = F.softmax(orig)
        for j in range(len(tfs_kept)):
            X_new = torch.clone(X_test).detach()
            for num_gene in range(len(genes_filtered)):
                X_new[:, cut[num_gene]+j] = 0
            
            perturb = model(X_new)
            perturb = F.softmax(perturb)
            results[i, :, j, :] = (perturb-orig).detach().cpu().numpy()
            results_orig[i, :, j, :] = orig.clone().detach().cpu().numpy()
            results_perturb[i, :, j, :] = perturb.detach().cpu().numpy()
        print("Done!")

    np.save(args.savepath+'processing/concat.'+args.model+'_TF_down.npy', results)
    np.save(args.savepath+'processing/concat.'+args.model+'_TF_down_orig.npy', results_orig)
    np.save(args.savepath+'processing/concat.'+args.model+'_TF_down_perturb.npy', results_perturb)


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
df_results.to_csv(args.savepath+'results/concat.'+args.model+'.TF_rank_downregulation.csv', index=False)
print("Done!")

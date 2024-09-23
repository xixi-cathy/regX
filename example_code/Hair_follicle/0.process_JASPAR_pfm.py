# This script describes how to convert the format of the PFM matrices from the .TRANSFAC files downloaded from the JASPAR database (https://jaspar.elixir.no/) to .npy files

from pyfaidx import Fasta
import numpy as np
import pandas as pd
import torch
from torch import nn
from numba import jit
import time
import os
import h5py
import math
os.chdir("/nfs/public/xixi/ref_genome")


motif_files = os.listdir('./JASPAR_motifs_pfm_mouse')

for i in range(len(motif_files)):
    motif_file = open('./JASPAR_motifs_pfm_mouse/'+motif_files[i])
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
    np.save("./JASPAR_motifs_pfm_mouse/pfm.np/"+motif+".npy", mat)
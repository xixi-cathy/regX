# regX
This repository stores the custom code for the paper "A mechanism-informed deep neural network enables prioritization of regulators that drive cell state transitions".

Users may run the scripts in each folder in a numbered order to reproduce the results in the paper. For example, "0.preprocess_HairFollicle.R.ipynb" and "0.process_JASPAR_pfm.py" are the first processing steps, and "1.nn_train_step1.py" is the second step.

# Software preparation
Prepare Python and R environment.

We used the following Python packages with python 3.10.4 under the Anaconda environment (conda 23.1.0):  
torch: 1.13.1+cu117; numpy: 1.23.4; scipy: 1.8.1; pandas: 1.4.1; scikit-learn: 1.1.3;  anndata:0.8.0; captum: 0.6.0; scanpy: 1.9.1; seaborn: 0.12.1; pyfaidx: 0.7.1; h5py: 3.7.0; verstack: 3.6.7.

We used the following R packages with R 4.2.2:  
dplyr: 1.1.0; ggplot2: 3.4.1; EnsDb.Hsapiens.v75: 2.99.0; EnsDb.Mmusculus.v79: 2.99.0; SeuratDisk: 0.0.0.9020; Seurat: 4.3.0; Signac: 1.9.0; Matrix: 1.6-1.1; DIRECTNET: 0.0.1.

We used the following R packages with R 4.1.3:   
SeuratWrappers: 0.3.1; monocle3: 1.3.1.

We also used PLINK software (v1.90b7.2 64-bit) on Windows PC.

# Data preparation
## T2D study
* 10x multiome data: download from the GEO database under accession number GSE200044 (https://ftp.ncbi.nlm.nih.gov/geo/series/GSE200nnn/GSE200044/suppl/GSE200044%5Fprocessed%5Fmatrix.tar.gz).
* Processed genotype data: download from the GEO database under accession number GSE170763 (https://ftp.ncbi.nlm.nih.gov/geo/series/GSE170nnn/GSE170763/suppl/GSE170763%5Fplink%5Ffiles.tar.gz).
* The PFM matrices of human motifs (.TRANSFAC files): download from the JASPAR database (https://jaspar.elixir.no/), or from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/JASPAR_motifs_pfm_homosapiens.zip).
* The hg19 reference genome: download from the UCSC genome browser (https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz).
* Processed genomic annotation file of hg19: download from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/hg19_geneanno.txt).
* Human PPI network: download from the STRING database (https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz).
* The genomic annotations of significant GWAS SNPs of T2D: download from the UCSC genome browser, or download the processed file from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/GWAS_T2D_hg19_UCSC.csv).

## Hair follicle study
* Processed SHARE-seq data: download from the scglue package (http://download.gao-lab.org/GLUE/dataset/Ma-2020-RNA.h5ad for RNA data and http://download.gao-lab.org/GLUE/dataset/Ma-2020-ATAC.h5ad for ATAC data).
* The PFM matrices of mouse motifs (.TRANSFAC files): download from the JASPAR database (https://jaspar.elixir.no/), or from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/JASPAR_motifs_pfm_mouse.zip).
* The mm10 reference genome: download from the UCSC genome browser (https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/latest/mm10.fa.gz).
* Processed genomic annotation file of mm10: download from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/mm10_geneanno.txt).
* GO relationship file "go-basic.obo": download from the GO database (http://purl.obolibrary.org/obo/go/go-basic.obo).
* GO annotation file "mgi.gaf": download from https://current.geneontology.org/annotations/mgi.gaf.gz.

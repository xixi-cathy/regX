# regX
This repository stores the custom code for the paper "A mechanism-informed deep neural network enables prioritization of regulators that drive cell state transitions".

# Contact
Xi Xi (xix19@mails.tsinghua.edu.cn)

# Software preparation
Prepare Python and R environment on a Linux system.

We use the following Python packages with python 3.10.4 under the Anaconda environment (conda 23.1.0):  
torch: 1.13.1+cu117; numpy: 1.23.4; scipy: 1.8.1; pandas: 1.4.1; scikit-learn: 1.1.3;  anndata:0.8.0; captum: 0.6.0; scanpy: 1.9.1; seaborn: 0.12.1; pyfaidx: 0.7.1; h5py: 3.7.0; verstack: 3.6.7.

We use the following R packages with R 4.2.2:  
dplyr: 1.1.0; ggplot2: 3.4.1; EnsDb.Hsapiens.v75: 2.99.0; EnsDb.Mmusculus.v79: 2.99.0; SeuratDisk: 0.0.0.9020; Seurat: 4.3.0; Signac: 1.9.0; Matrix: 1.6-1.1; DIRECTNET: 0.0.1.

We use the following R packages with R 4.1.3:   
SeuratWrappers: 0.3.1; monocle3: 1.3.1.

We also used PLINK software (v1.90b7.2 64-bit) on a Windows PC.

# Environment setup
We strongly recommend you install Anaconda3, where we use Python and R.

To set up your Python environment:
```
conda create --name pyenv python=3.10
conda activate pyenv
pip install -r requirements.txt
```

To set up your R environment:
```
conda create --name Renv python=3.10
conda activate Renv
conda install r-base
```
And install R packages listed in the "Software preparation" section.

Also, install jupyter in your conda environment to run the notebook files:
```
conda install jupyter
```
To run a Jupyter Notebook with R, please follow instructions [here](https://izoda.github.io/site/anaconda/r-jupyter-notebook/).

The installation and environment setup normally takes about 2 hours.

# Usage examples
We provide two examples to demonstrate the usage of regX. Users may run the scripts in each example folder in a numbered order.

## Hair follicle example

### Data preparation
* Processed SHARE-seq data: download from the scglue package (http://download.gao-lab.org/GLUE/dataset/Ma-2020-RNA.h5ad for RNA data and http://download.gao-lab.org/GLUE/dataset/Ma-2020-ATAC.h5ad for ATAC data).
* The PFM matrices of mouse motifs (.TRANSFAC files): download from the JASPAR database (https://jaspar.elixir.no/), or from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/JASPAR_motifs_pfm_mouse.zip).
* The mm10 reference genome: download from the UCSC genome browser (https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/latest/mm10.fa.gz).
* Processed genomic annotation file of mm10: download from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/mm10_geneanno.txt).
* GO relationship file "go-basic.obo": download from the GO database (http://purl.obolibrary.org/obo/go/go-basic.obo).
* GO annotation file "mgi.gaf": download from https://current.geneontology.org/annotations/mgi.gaf.gz.

### Step-by-step workflow
The custom code for the T2D example was stored in the ".examples/Hair_follicle" folder. Users need to modify the working directories according to their date deposition before running the scripts. We provide the output files in each step [here](https://zenodo.org/records/11607943?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjIwN2NhYjhlLThmNGQtNDllOC1iYWI2LTVmNThlZTVjNzkyMiIsImRhdGEiOnt9LCJyYW5kb20iOiIzM2I1YmYzNGQzNTViYjg3MGZlZDY4MDM3YjJhMmY1MyJ9.HBiLzhKg0-Hfnr7TrinVhhKuk_JkC4X5b4QEs3i7Fuebw0zQAJM8CVVew_7SqZPf6RYDq0gjBRayt8s8XL3kIQ).

1. **Process the single-cell multi-omics dataset.**
  
   **Expected outputs**: Pseudo-bulk samples and labels for model training.   
   This step takes about an hour.
   
2. **Process the JASPAR PFM matrix files.**
   
   **Expected outputs**: .npy files storing the PFM matrices of motifs.   
   This step takes about 2 minutes.
   
3. **Process the gene-GO links and GO graph embedded in the model.**

   **Expected outputs**: gene-GO and GO-GO interactions.   
   This step takes about 10 minutes.
   
4. **The first step of training for regX.**
   
   **Expected outputs**: the trained gene-level models and performance (each model was trained for three times under random seeds).   
   This step takes about 2-3 hours on the GPU.

5. **Extract the TAM matrices.**

   **Expected outputs**: The input TAM matrices, output labels and metadata for regX.   
   This step takes about an hour.
   
6. **The second step of training for regX.**

   **Expected outputs**: parameters of the trained model (the model was trained for 10 times under random seeds).    
   This step takes about 2 hours on gpu.

7. **Model interpretation: in-silico perturbation.**

   **Expected outputs**: state-transitional probabilities before and after in-silico perturbation of TFs.   
   This step takes about 5 minutes on the GPU.

8. **Prioritization and visualization of pdTFs.**

   **Expected outputs**: state transitional graphs.   
   This step takes about 2 minutes. The output files were provided in Extended Data Figures.
   

## T2D example
### Data preparation
* 10x multiome data: download from the GEO database under accession number GSE200044 (https://ftp.ncbi.nlm.nih.gov/geo/series/GSE200nnn/GSE200044/suppl/GSE200044%5Fprocessed%5Fmatrix.tar.gz).
* Processed genotype data: download from the GEO database under accession number GSE170763 (https://ftp.ncbi.nlm.nih.gov/geo/series/GSE170nnn/GSE170763/suppl/GSE170763%5Fplink%5Ffiles.tar.gz).
* The PFM matrices of human motifs (.TRANSFAC files): download from the JASPAR database (https://jaspar.elixir.no/), or from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/JASPAR_motifs_pfm_homosapiens.zip).
* The hg19 reference genome: download from the UCSC genome browser (https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz).
* Processed genomic annotation file of hg19: download from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/hg19_geneanno.txt).
* Human PPI network: download from the STRING database (https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz).
* The genomic annotations of significant GWAS SNPs of T2D: download from the UCSC genome browser, or download the processed file from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/GWAS_T2D_hg19_UCSC.csv).

### Step-by-step workflow
The custom code for the T2D example was stored in the ".examples/T2D" folder. Users need to modify the working directories according to their date deposition before running the scripts. We provide part of the output files [here](https://zenodo.org/records/11608076?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVhMzk5Nzk0LWQ0MDYtNDE3Yi1hNjZhLThjMmJhMDU0NjgyMyIsImRhdGEiOnt9LCJyYW5kb20iOiI1YjQ2Nzc3OTQ1OTNkNzkzMTU3ODU5YjBmZDNkMDdkNSJ9.Ha_9ZGH5wEOaiSu3LTRlqPgcbkQUVzrUN8DPkWeGRAZ3LArNlcRgDrxdESyXKpv7ag81twiWpz9TWsnTZwJMkg). The rest files are relatively large, and can be provided upon request.

1. **Process the single cell multi-omics dataset.**
  
   **Expected outputs**: DE genes of cells from normal, non-diabetic and diabetic donors; pseudo-bulk samples and labels for model training.   
   This step takes about an hour.
   
2. **Process the JASPAR PFM matrix files.**
   
   **Expected outputs**: .npy files storing the PFM matrices of motifs.   
   This step takes about 2 minutes.
   
3. **The first step of training for regX.**
   
   **Expected outputs**: the trained gene-level models (each model was trained for three times under random seeds).   
   **Note:** This step takes more than 12 hours by running four scripts (1/4 genes in each script) simultaneously on the GPU.

4. **Extract the TAM matrices.**

   **Expected outputs**: The input TAM matrices, output labels and metadata for regX.   
   This step takes about an hour.
   
5. **Process the PPI network embedded in the model.**

   **Expected outputs**: filtered PPI links.   
   This step takes about 10 minutes.
   
6. **The second step of training for regX.**

   **Expected outputs**: parameters of the trained model (the model was trained for 10 times under random seeds).    
   This step takes about 2 hours on gpu.

7. **Model interpretation: in-silico perturbation.**

   **Expected outputs**: state-transitional probabilities before and after in-silico perturbation of TFs and cCREs.   
   **Note:** This step takes about 12 hours on the GPU (mainly because perturbing the cCREs is time-consuming).

8. **Prioritization and visualization of pdTFs and pdCREs.**

   **Expected outputs**: state transitional graphs.   
   This step takes about 2 minutes. The output files were provided in Extended Data Figures.
   
9. **Model interpretation: prioritize target genes of a pdTF.**
   
   **Expected outputs**: prioritization list of the target genes.   
   This step takes about 5 minutes. The output files were provided in Supplementary tables.

# Instructions for use on your own data
regX is a relatively flexible deep learning framework. The prediction tasks and model structures can be flexibly designed according to the application scenario. In general, you may refer to the instructions and workflows demonstrated in the two examples. You may also contact us for more technical support.

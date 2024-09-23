# regX
This repository stores the custom code for the paper "A mechanism-informed deep neural network enables prioritization of regulators that drive cell state transitions".

# Contact
Xi Xi (xix19@mails.tsinghua.edu.cn)


# Environment setup
We highly recommend installing Anaconda3, which supports isolated Python and R environments.

* **Python environment:**
   ```
   conda create --name regX python=3.10
   conda activate regX
   pip install -r requirements.txt
   ```

* **R environment** (optional, needed only for reproducing the two usage examples):
   ```
   conda env create -f environment.yml
   conda activate regXr
   ```
   Then, install the remaining packages in R:
   ```
   install.packages("BiocManager")
   BiocManager::install(c("GenomeInfoDbData", "cicero", "chromVAR", "motifmatchr"))
   
   devtools::install_github(c("zhanglhbioinfor/DIRECT-NET", "mojaveazure/seurat-disk", "satijalab/seurat-wrappers@community-vignette"))
   ```

# Usage on your own data
## Data preparation
1. Download files in the data folder, and unzip them. Download the custom code.
2. (Compulsory) Replace the "rna.csv", "atac.csv", "label.csv", and "genes.txt" files with your own data. The file names and formats should be the same as what we provided.

   "rna.csv" contains the normalized gene expression levels of pseudo-bulk samples. "atac.csv" contains the normalized chromatin accessibilities of corresponding samples. "label.csv" contains the cell state labels of these samples. "genes.txt" contains a list of target genes to be included in the hidden layer of regX (better not to exceed 300 genes, or the computational cost will be very high).

   For pre-processing single-cell multi-omics data and generating pseudo-bulk samples, you may refer to our [example code](example_code/T2D/0.preprocess_T2D.R.ipynb) for more detail.
   
4. (Optional) If you wish to integrate GO functions into regX, replace the "go_filtered.txt" and "goa_filtered.txt" files with the GO graph and related GO annotations that you selected.

   You may refer to our [example code](example_code/Hair_follicle/0.process_GO) for more detail.
   
   Genes in the "genes.txt" file should be consistent with those in the GO graph that you selected.
   
5. (Optional) If you wish to integrate PPIs into regX, download PPI networks from the STRING database and place them in the same directory as the data files mentioned above.
   ```
   filepath=/data1/xixi/regX/code_test/
   cd $filepath
   ```
   For human: 
   ```
   wget https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
   wget https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz
   ```
   For mouse:
   ```
   wget https://stringdb-downloads.org/download/protein.links.v12.0/10090.protein.links.v12.0.txt.gz
   wget https://stringdb-downloads.org/download/protein.info.v12.0/10090.protein.info.v12.0.txt.gz
   ```
   Then, unzip these files:
   ```
   gunzip *.gz
   ```
## Step-by-step workflow
1. Basic setup

   Before starting, ensure that the Python environment 'regX' is properly set up. Then, activate the environment, define the necessary variables, and navigate to the code directory:
   ```
   conda activate regX
   codepath=/home/xixi/scRegulate/code_upload/custom_code     # Folder path where you stored the custom code 
   filepath=/data1/xixi/regX/code_test/                       # Folder path where you stored the data
   savepath=/data1/xixi/regX/code_test/                       # Folder path where you decided to save the processing and result files
   species=mouse                                              # Species of your data. We only support "human" and "mouse" for now.
   refgenome=mm10                                             # Reference genome. We only support "hg19" and "mm10" for now.
   device=cuda:3                                              # Device for model training and interpretation. We highly recommend using GPU.
   cd $codepath
   ```

2. First step of training: learn the input feature matrix TAM

   (Runtime on the demo data: ~ 2 hours)
   ```
   python learn_W.py --filepath $filepath --savepath $savepath --ref $refgenome --device $device
   python generate_TAM.py --filepath $filepath --savepath $savepath --ref $refgenome --device $device
   ```

3. (Optional) Extract a PPI sub-network:

   (Runtime on the demo data: < 1 minute)
   ```
   python process_ppi.py --filepath $filepath --savepath $savepath --species $species
   ```
   You may skip this step if using a GO graph.

4. Second step of training: train the regX model

   (Runtime on the demo data: < 10 minutes)
   ```
   python run_regX.py --filepath $filepath --savepath $savepath --ref $refgenome --device $device --model GCN --replicates 5 --batchsize 256 --lr 0.001
   ```
   You may freely choose a GCN, SAGEConv, or GAT model according to your needs. We recommend using GCN or SAGEConv for an undirected PPI network, and GAT for a directed GO graph. The performance of these structures on our demo data is relatively close, and can be found [here](figures/performance_comparison.png).

5. Prioritize TFs and cCREs

   **For TF prioritization:**
   
   (Runtime on the demo data: < 5 minutes)
   ```
   python prioritize_TF.py --filepath $filepath --savepath $savepath --ref $refgenome --device $device --model GCN --batchsize 256 --lr 0.001 --top 5
   ```
   In silico upregulation and downregulation of each TF will be performed, respectively. TFs in the "...TF_rank_upregulation.csv" file mean that the upregulation of these TFs will induce corresponding cell state transitions, and TFs in the "...TF_rank_downregulation.csv" file mean that the downregulation of these TFs will induce corresponding cell state transitions.
   
   You may assign the number of top-ranked TFs between every two states with "--top".
   
   
   **For cCRE prioritization:**

   (Runtime on the demo data: ~ 2 hours)
   ```
   python prioritize_cCRE.py --filepath $filepath --savepath $savepath --ref $refgenome --device $device --model GCN --batchsize 256 --lr 0.001 --top 100
   ```
   In silico opening and closing of each cCRE will be performed, respectively. You may assign the number of top-ranked cCREs between every two states with "--top".

The trained models, top-ranked regulators, and cell state transition graphs can be found in the "results" folder. Some intermediate files can be found in the "processing" folder.

**In addition:**

&emsp;&emsp;We also offer a lightweight version of the regX model (regX-light), which replaces the TAM with a concatenated TF and cCRE feature vector. This version requires only one training step and can be used for TF prioritization, but not for cCRE prioritization (see the discussion section in our paper for more details). While we still recommend the TAM-based approach, regX-light could be a suitable option for those with limited computational resources who are primarily interested in identifying driver TFs. To use regX-light, you can skip steps 2-5 and run the following command:

(Runtime on the demo data: < 10 minutes)
```
python run_regX-light.py --filepath $filepath --savepath $savepath --ref $refgenome --device $device --model GCN --replicates 5 --batchsize 256 --lr 0.001 --top 5
```


For more usage, please refer to the two examples below.

# Example cases
We provide two examples to demonstrate how to use regX. Users can run the scripts in each example folder in the specified numerical order.

## Hair follicle example

### Data preparation
* Processed SHARE-seq data: download from the scglue package (http://download.gao-lab.org/GLUE/dataset/Ma-2020-RNA.h5ad for RNA data and http://download.gao-lab.org/GLUE/dataset/Ma-2020-ATAC.h5ad for ATAC data).
* The PFM matrices of mouse motifs (.TRANSFAC files): download from the JASPAR database (https://jaspar.elixir.no/), or from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/JASPAR_motifs_pfm_mouse.zip).
* The mm10 reference genome: download from the UCSC genome browser (https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/latest/mm10.fa.gz).
* Processed genomic annotation file of mm10: download from the "data" folder (https://github.com/xixi-cathy/regX/tree/main/data/mm10_geneanno.txt).
* GO relationship file "go-basic.obo": download from the GO database (http://purl.obolibrary.org/obo/go/go-basic.obo).
* GO annotation file "mgi.gaf": download from https://current.geneontology.org/annotations/mgi.gaf.gz.

### Step-by-step workflow
The custom code for the T2D example is stored in the "example_code/Hair_follicle" folder. Users need to adjust the working directories based on their date locations before running the scripts. We provide the output files in each step [here](https://zenodo.org/records/11607943?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjIwN2NhYjhlLThmNGQtNDllOC1iYWI2LTVmNThlZTVjNzkyMiIsImRhdGEiOnt9LCJyYW5kb20iOiIzM2I1YmYzNGQzNTViYjg3MGZlZDY4MDM3YjJhMmY1MyJ9.HBiLzhKg0-Hfnr7TrinVhhKuk_JkC4X5b4QEs3i7Fuebw0zQAJM8CVVew_7SqZPf6RYDq0gjBRayt8s8XL3kIQ).

1. **[Process the single-cell multi-omics dataset.](example_code/Hair_follicle/0.preprocess_HairFollicle.R.ipynb)**
  
   **Expected outputs**: Pseudo-bulk samples and labels for model training.   
   This step takes about an hour.
   
2. **[Process the JASPAR PFM matrix files.](example_code/Hair_follicle/0.process_JASPAR_pfm.py)**
   
   **Expected outputs**: .npy files storing the PFM matrices of motifs.   
   This step takes about 2 minutes.
   
3. **[Process the gene-GO links and GO graph embedded in the model.](example_code/Hair_follicle/0.process_GO)**

   **Expected outputs**: gene-GO and GO-GO interactions.   
   This step takes about 10 minutes.
   
4. **[The first step of training for regX.](example_code/Hair_follicle/1.nn_train_step1.py)**
   
   **Expected outputs**: the trained gene-level models and performance (each model was trained three times under random seeds).   
   This step takes about 2-3 hours on the GPU.

5. **[Extract the TAM matrices.](example_code/Hair_follicle/2.generate_TAM.py.ipynb)**

   **Expected outputs**: The input TAM matrices, output labels, and metadata for regX.   
   This step takes about an hour.
   
6. **[The second step of training for regX.](example_code/Hair_follicle/3.nn_train_step2.py)**

   **Expected outputs**: parameters of the trained model (the model was trained 10 times under random seeds).    
   This step takes about 2 hours on GPU.

7. **[Model interpretation: in-silico perturbation.](example_code/Hair_follicle/4.in-silico_perturbation.py.ipynb)**

   **Expected outputs**: state-transitional probabilities before and after in-silico perturbation of TFs.   
   This step takes about 5 minutes on the GPU.

8. **[Prioritization and visualization of pdTFs.](example_code/Hair_follicle/5.prioritization_and_visualization.py.ipynb)**

   **Expected outputs**: state transitional graphs.   
   This step takes about 2 minutes. The output files are provided in the Supplementary Figures.
   

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
The custom code for the T2D example is stored in the "example_code/T2D" folder. Users need to adjust the working directories based on their date locations before running the scripts. We provide part of the output files [here](https://zenodo.org/records/11608076?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVhMzk5Nzk0LWQ0MDYtNDE3Yi1hNjZhLThjMmJhMDU0NjgyMyIsImRhdGEiOnt9LCJyYW5kb20iOiI1YjQ2Nzc3OTQ1OTNkNzkzMTU3ODU5YjBmZDNkMDdkNSJ9.Ha_9ZGH5wEOaiSu3LTRlqPgcbkQUVzrUN8DPkWeGRAZ3LArNlcRgDrxdESyXKpv7ag81twiWpz9TWsnTZwJMkg). The remaining files are relatively large and can be provided upon request.

1. **[Process the single cell multi-omics dataset.](example_code/T2D/0.preprocess_T2D.R.ipynb)**
  
   **Expected outputs**: DE genes of cells from normal, non-diabetic, and diabetic donors; pseudo-bulk samples and labels for model training.   
   This step takes about an hour.
   
2. **[Process the JASPAR PFM matrix files.](example_code/T2D/0.process_JASPAR_pfm.py)**
   
   **Expected outputs**: .npy files storing the PFM matrices of motifs.   
   This step takes about 2 minutes.
   
3. **[The first step of training for regX.](example_code/T2D/1.nn_train_step1.py)**
   
   **Expected outputs**: the trained gene-level models (each model was trained three times under random seeds).   
   **Note:** This step takes more than 12 hours by running four scripts (1/4 genes in each script) simultaneously on the GPU.

4. **[Extract the TAM matrices.](example_code/T2D/2.generate_TAM.py.ipynb)**

   **Expected outputs**: The input TAM matrices, output labels, and metadata for regX.   
   This step takes about an hour.
   
5. **[Process the PPI network embedded in the model.](example_code/T2D/2.process_STRING_ppi.R.ipynb)**

   **Expected outputs**: filtered PPI links.   
   This step takes about 10 minutes.
   
6. **[The second step of training for regX.](example_code/T2D/3.nn_train_step2.py)**

   **Expected outputs**: parameters of the trained model (the model was trained 10 times under random seeds).    
   This step takes about 2 hours on GPU.

7. **[Model interpretation: in-silico perturbation.](example_code/T2D/4.in-silico_perturbation.py.ipynb)**

   **Expected outputs**: state-transitional probabilities before and after in-silico perturbation of TFs and cCREs.   
   **Note:** This step takes about 12 hours on the GPU (because perturbing the cCREs is time-consuming).

8. **[Prioritization and visualization of pdTFs and pdCREs.](example_code/T2D/5.prioritization_and_visualization.py.ipynb)**

   **Expected outputs**: state transitional graphs.   
   This step takes about 2 minutes. The output files are provided in the Supplementary Figures.
   
9. **[Model interpretation: prioritize target genes of a pdTF.](example_code/T2D/6.TGs_of_pdTFs.py.ipynb)**
   
   **Expected outputs**: prioritization list of the target genes.   
   This step takes about 5 minutes. The output files are provided in the Supplementary tables.

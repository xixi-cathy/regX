import os
import sys
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, default='./', help='Path of the data files (all data files should be saved under a same directory)')
parser.add_argument('--savepath', type=str, default='./', help='Path of the result folder')
parser.add_argument('--species', type=str, default='human', help='human or mouse')

args = parser.parse_args()

filepath = args.filepath
savepath = args.savepath
species = args.species

os.chdir(filepath)

# Processing PPI
print("Processing STRING PPI...")
if species == 'human':
    ppi = pd.read_csv(os.path.join(filepath, '9606.protein.links.v12.0.txt'), sep=' ')
    info = pd.read_csv(os.path.join(filepath, '9606.protein.info.v12.0.txt'), sep='\t')
elif species == 'mouse':
    ppi = pd.read_csv(os.path.join(filepath, '10090.protein.links.v12.0.txt'), sep=' ')
    info = pd.read_csv(os.path.join(filepath, '10090.protein.info.v12.0.txt'), sep='\t')
else:
    sys.exit(f'Species {species} is not supported. Please try human or mouse.')

genenames = pd.Series(info['preferred_name'].values, index=info['#string_protein_id']).to_dict()
ppi['protein1'] = ppi['protein1'].map(genenames)
ppi['protein2'] = ppi['protein2'].map(genenames)
print("Done!")

### Extracting sub-networks
print("\nExtracting sub-network...\n")
genes = pd.read_csv(os.path.join(savepath, 'processing/pearson.testset.txt'), sep='\t', header=None)[0].to_list()
print(f"{len(genes)} genes in total")
inds = pd.Series(range(len(genes)), index=genes).to_dict()

edges = ppi[(ppi['protein1'].isin(genes)) & (ppi['protein2'].isin(genes))][['protein1', 'protein2']].drop_duplicates()
edges['protein1'] = edges['protein1'].map(inds)
edges['protein2'] = edges['protein2'].map(inds)
edges = edges.drop_duplicates()

edges.to_csv(os.path.join(savepath, 'processing/ppi_filtered.txt'), sep='\t', index=False)
print("Done!\n")
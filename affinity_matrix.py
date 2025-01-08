import pandas as pd
import numpy as np
import snf
import os
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

tcga_project = 'BRCA'

# Caricamento dei dati di espressione (datExpr_<omics>)
print("Loading data...")
datExpr_DNAm = f"../MOGDx/data/TCGA/{tcga_project}/raw/datExpr_DNAm.csv"
datExpr_miRNA = f"../MOGDx/data/TCGA/{tcga_project}/raw/datExpr_miRNA.csv"
datExpr_mRNA = f"../MOGDx/data/TCGA/{tcga_project}/raw/datExpr_mRNA.csv"

df_DNAm = pd.read_csv(datExpr_DNAm, index_col=0)
df_miRNA = pd.read_csv(datExpr_miRNA, index_col=0)
df_mRNA = pd.read_csv(datExpr_mRNA, index_col=0)

# Carica gli indici/nomi dei migliori geni/sitiCPG
cpg_sites_DNAm = np.loadtxt(f"../MOGDx/data/TCGA/{tcga_project}/raw/cpg_sites_DNAm.txt", dtype=str)
cpg_sites_DNAm = [cpg.strip('"').strip() for cpg in cpg_sites_DNAm]
top_genes_miRNA = np.loadtxt(f"../MOGDx/data/TCGA/{tcga_project}/raw/top_genes_miRNA.txt", dtype=int) - 1
top_genes_mRNA = np.loadtxt(f"../MOGDx/data/TCGA/{tcga_project}/raw/top_genes_mRNA.txt", dtype=int) - 1

# Filtra il dataframe mantenendo solo i migliori geni/sitiCPG
df_DNAm_cpgs = df_DNAm.loc[:, cpg_sites_DNAm]
df_miRNA_tg = df_miRNA.iloc[top_genes_miRNA, :]
df_mRNA_tg = df_mRNA.iloc[top_genes_mRNA, :]

# Transporre i dataframe per avere i pazienti ogni riga e le features come colonne
df_miRNA_tg_transpose = df_miRNA_tg.transpose()
df_mRNA_tg_transpose = df_mRNA_tg.transpose()

print("Shape of DNAm, miRNA, mRNA:")
print(df_DNAm_cpgs.shape, df_miRNA_tg_transpose.shape, df_mRNA_tg_transpose.shape)

# Trova gli indici comuni tra i dataframe di ogni omica
common_indices = np.intersect1d(np.intersect1d(df_DNAm_cpgs.index, df_miRNA_tg_transpose.index), df_mRNA_tg_transpose.index)
print("Common indices:", len(common_indices)) #test
print("Common indices:", common_indices) #test


# Filtra il dataframe mantenendo solo i campioni comuni
df_DNAm_cpgs = df_DNAm_cpgs.loc[common_indices]
df_miRNA_tg_transpose = df_miRNA_tg_transpose.loc[common_indices]
df_mRNA_tg_transpose = df_mRNA_tg_transpose.loc[common_indices]

# Calcolo delle matrici di affinità
affinity_networks_DNAm = snf.make_affinity(df_DNAm_cpgs, metric='euclidean', K=20, mu=0.5, normalize=True)
print("DNAm affinity networks created")
affinity_networks_miRNA = snf.make_affinity(df_miRNA_tg_transpose, metric='euclidean', K=20, mu=0.5, normalize=True)
print("miRNA affinity networks created")
affinity_networks_mRNA = snf.make_affinity(df_mRNA_tg_transpose, metric='euclidean', K=20, mu=0.5, normalize=True)
print("mRNA affinity networks created")

directory = f'affinity_matrices/{tcga_project}'
if not os.path.exists(directory):
    os.makedirs(directory)

# Salva le matrici di affinità in npy
np.save(f'affinity_matrices/{tcga_project}/fused_affinity_matrix_DNAm.npy', affinity_networks_DNAm)
np.save(f'affinity_matrices/{tcga_project}/fused_affinity_matrix_miRNA.npy', affinity_networks_miRNA)
np.save(f'affinity_matrices/{tcga_project}/fused_affinity_matrix_mRNA.npy', affinity_networks_mRNA)

# Calcolo della matrice di affinità fusa
fused_affinity_matrix = snf.snf([affinity_networks_DNAm, affinity_networks_miRNA, affinity_networks_mRNA], K=20)
print(f"Fused affinity networks created for {tcga_project}, shape: {fused_affinity_matrix.shape}")

# Salva la matrice di affinità fusa in npy
np.save(f'affinity_matrices/{tcga_project}/fused_affinity_matrix.npy', fused_affinity_matrix)

# Aggiungi gli id dei pazienti come etichette di riga e colonna
fused_affinity_matrix_df = pd.DataFrame(fused_affinity_matrix, index=common_indices, columns=common_indices)

# Salva la matrice di affinità fusa con etichette in CSV
fused_affinity_matrix_df.to_csv(f'affinity_matrices/{tcga_project}/fused_affinity_matrix_with_id.csv')

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

# Filtra il dataframe mantenendo solo i campioni comuni
df_DNAm_cpgs = df_DNAm_cpgs.loc[common_indices]
df_miRNA_tg_transpose = df_miRNA_tg_transpose.loc[common_indices]
df_mRNA_tg_transpose = df_mRNA_tg_transpose.loc[common_indices]

# def calculate_n_components(df, proportion=1):
#     """
#     Calcola il numero di componenti da mantenere, tenendo conto che n_components non deve superare il numero di righe.
    
#     df: Il dataframe con i dati.
#     proportion: La proporzione delle colonne da mantenere.
#     return: Il numero di componenti da usare nell'RSVD.
#     """
#     n_components = min(int(df.shape[0] * proportion), int(df.shape[1] * proportion)) - 1  # Calcola il numero di componenti
#     return n_components

# def apply_rsvd(data_matrix, n_components):
#     """
#     Applica RSVD su un set di dati.

#     data_matrix: DataFrame con dati multi-omici.
#     n_components: Numero di componenti da mantenere dopo la riduzione dimensionale.
#     return: Il dataframe con dimensionalità ridotta. 
#     """
#     # Estrae la matrice dai dati
#     X = np.array(data_matrix.iloc[:, 1:], dtype=np.float64)
    
#     # Applica RSVD
#     U, Sigma, VT = randomized_svd(X, n_components=n_components, random_state=42)
    
#     # Proietta i dati nello spazio ridotto
#     X_reduced = U @ np.diag(Sigma)
    
#     # Restituisce il risultato come DataFrame
#     return pd.DataFrame(X_reduced, index=data_matrix.index)

# n_components_mirna = calculate_n_components(df_mirna, proportion=0.2)  # 20% per miRNA
# n_components_rnaseq = calculate_n_components(df_rnaseq, proportion=0.5)  # 50% per RNASeq
# n_components_rppa = calculate_n_components(df_rppa, proportion=0.2)  # 20% per RPPA

# # Applichiamo RSVD su ogni tipo di dato
# df_mirna_reduced = apply_rsvd(df_mirna, n_components_mirna)
# df_rnaseq_reduced = apply_rsvd(df_rnaseq, n_components_rnaseq)
# df_rppa_reduced = apply_rsvd(df_rppa, n_components_rppa)

# print("Shape dopo RSVD:")
# print(df_mirna_reduced.shape, df_rnaseq_reduced.shape, df_rppa_reduced.shape) #, df_methy_reduced.shape)

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

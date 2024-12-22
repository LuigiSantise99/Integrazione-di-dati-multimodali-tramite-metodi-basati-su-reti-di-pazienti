import pandas as pd
import numpy as np
import snf
import os
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

tcga_project = 'BRCA'

# Caricamento dei dati di espressione (datExpr_miRNA)
datExpr_miRNA = f"../MOGDx/data/TCGA/{tcga_project}/raw/datExpr_miRNA.csv"
df_miRNA = pd.read_csv(datExpr_miRNA, index_col=0)

# Carica gli indici dei geni da R e converti a base-0
top_genes = np.loadtxt(f"../MOGDx/data/TCGA/{tcga_project}/raw/top_genes.txt", dtype=int) - 1

# Filtra il dataframe mantenendo solo le righe corrispondenti agli indici
filtered_df = df_miRNA.iloc[top_genes, :]

# Verifica la dimensione del dataframe filtrato e del numero di indici
print("Shape prima di RSVD:")
print(filtered_df.shape)
print(f"Numero di indici nel file top_genes.txt: {len(top_genes)}")
print(f"Numero di righe nel dataframe filtrato: {filtered_df.shape[0]}")

# Controlla i nomi dei geni selezionati
selected_genes = df_miRNA.index[top_genes]
print("Ecco alcuni dei geni selezionati:")
print(selected_genes[:10])  # Mostra i primi 10 geni selezionati

# Confronta con i valori del file top_genes.txt
with open(f"../MOGDx/data/TCGA/{tcga_project}/raw/top_genes.txt", 'r') as f:
    top_genes_in_file = f.readlines()

# Rimuovi spazi bianchi e converte gli indici in base-1 (per il confronto con il file R)
top_genes_in_file = [str(int(gene.strip()) + 1) for gene in top_genes_in_file]  # Converte a base-1

# Verifica che i primi 10 geni nel file siano gli stessi estratti
print("Confronto dei primi 10 geni (base-1):")
print(top_genes_in_file[:10])
print("Confronto con i geni selezionati (base-1):")
print(selected_genes[:10])

# Verifica se c'è qualche differenza tra i geni estratti e quelli nel file
matching_genes = np.isin(selected_genes, top_genes_in_file)
print("Verifica della corrispondenza tra i geni selezionati e quelli nel file:")
print(f"Tutti i geni selezionati corrispondono ai geni nel file: {matching_genes.all()}")

# Salvataggio del dataframe filtrato per una verifica visiva
filtered_df.to_csv(f"filtered_miRNA_{tcga_project}.csv")
print("File CSV dei top genes salvato come 'filtered_miRNA_{tcga_project}.csv'")

# print(df_DNAm.shape, df_miRNA.shape, df_mRNA.shape)

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
# #n_components_methy = calculate_n_components(df_methy, proportion=0.05)  # 5% per Methy

# # Applichiamo RSVD su ogni tipo di dato
# df_mirna_reduced = apply_rsvd(df_mirna, n_components_mirna)
# df_rnaseq_reduced = apply_rsvd(df_rnaseq, n_components_rnaseq)
# df_rppa_reduced = apply_rsvd(df_rppa, n_components_rppa)
# #df_methy_reduced = apply_rsvd(df_methy, n_components_methy)

# print("Shape dopo RSVD:")
# print(df_mirna_reduced.shape, df_rnaseq_reduced.shape, df_rppa_reduced.shape) #, df_methy_reduced.shape)

# # Calcolo delle matrici di affinità
# affinity_networks_mirna = snf.make_affinity(df_mirna_reduced, metric='euclidean', K=20, mu=0.5, normalize=True)
# print("Mirna data affinity networks created")
# affinity_networks_rnaseq = snf.make_affinity(df_rnaseq_reduced, metric='euclidean', K=20, mu=0.5, normalize=True)
# print("Rnaseq data affinity networks created")
# affinity_networks_rppa = snf.make_affinity(df_rppa_reduced, metric='euclidean', K=20, mu=0.5, normalize=True)
# print("Rppa data affinity networks created")
# #affinity_networks_methy = snf.make_affinity(df_methy_reduced, metric='euclidean', K=20, mu=0.5, normalize=True)
# #print("Methy data affinity networks created")

# directory = f'{cancer_type}'
# if not os.path.exists(directory):
#     os.makedirs(directory)

# # Salva le matrici di affinità in npy e png
# np.save(f'{cancer_type}/fused_affinity_matrix_mirna.npy', affinity_networks_mirna)
# np.save(f'{cancer_type}/fused_affinity_matrix_rnaseq.npy', affinity_networks_rnaseq)
# np.save(f'{cancer_type}/fused_affinity_matrix_rppa.npy', affinity_networks_rppa)
# #np.save(f'{cancer_type}/fused_affinity_matrix_methy.npy', affinity_networks_methy)

# # Calcolo della matrice di affinità fuse
# fused_affinity = snf.snf([affinity_networks_mirna, affinity_networks_rnaseq, affinity_networks_rppa], K=20) #, affinity_networks_methy]
# print(f"Fused affinity networks created for {cancer_type}")

# # Salva la matrice di affinità fusa in npy e png
# np.save(f'{cancer_type}/fused_affinity_matrix.npy', fused_affinity)

# plt.imshow(fused_affinity, cmap='viridis', interpolation='nearest')
# plt.colorbar()
# plt.title(f'Fused Affinity Matrix - {cancer_type}')
# plt.savefig(f'{cancer_type}/fused_affinity_matrix_plot.png')  # Salva il plot come immagine

# '''
# # Carica la matrice di affinità fusa e la stampa 
# loaded_matrix = np.load(f'{cancer_type}/fused_affinity_matrix.npy')
# print(loaded_matrix.shape)
# print(loaded_matrix[:5])  # Controlla i primi 5 valori
# '''
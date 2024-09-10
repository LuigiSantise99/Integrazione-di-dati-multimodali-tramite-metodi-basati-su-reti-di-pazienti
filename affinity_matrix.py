import pandas as pd
import numpy as np
import snf
import os
# from sklearn.cluster import spectral_clustering, KMeans
# from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

cancer_type = 'BLCA2'

# Carica i dati da un file Excel
data_type_methy = f"dataset/cancer/{cancer_type}_Methy.csv"
data_type_mirna = f"dataset/cancer/{cancer_type}_miRNA.csv"
data_type_rnaseq = f"dataset/cancer/{cancer_type}_RNASeq.csv"
data_type_rppa = f"dataset/cancer/{cancer_type}_RPPA.csv"
df_rppa = pd.read_csv(data_type_rppa)
df_mirna = pd.read_csv(data_type_mirna)
df_rnaseq = pd.read_csv(data_type_rnaseq)
df_methy = pd.read_csv(data_type_methy, nrows=2) # ci mette molto tempo a caricare il file intero

print(df_mirna.shape, df_rnaseq.shape, df_rppa.shape)

def apply_rsvd(data_matrix, n_components=50):
    """
    Applica RSVD su un set di dati.
    data_matrix: DataFrame con dati multi-omici
    n_components: Numero di componenti da mantenere dopo la riduzione dimensionale
    """
    # Estrae la matrice dai dati
    X = np.array(data_matrix.iloc[:, 1:], dtype=np.float64)
    
    # Applica RSVD
    U, Sigma, VT = randomized_svd(X, n_components=n_components, random_state=42)
    
    # Proietta i dati nello spazio ridotto
    X_reduced = U @ np.diag(Sigma)
    
    # Restituisce il risultato come DataFrame
    return pd.DataFrame(X_reduced, index=data_matrix.index)

# Applichiamo RSVD su ogni tipo di dato
df_mirna_reduced = apply_rsvd(df_mirna)
df_rnaseq_reduced = apply_rsvd(df_rnaseq)
df_rppa_reduced = apply_rsvd(df_rppa)
# blca_methy_reduced = apply_rsvd(df_methy)

print(df_mirna_reduced.shape, df_rnaseq_reduced.shape, df_rppa_reduced.shape)

'''
# Estrai i dati e le etichette
data_columns_mirna = []
data_columns_methy = []
data_columns_exp = []
# colonne corrispondenti ai tuoi dati
# label_column = 'Death'  # colonna corrispondente alle etichette

data_mirna = df_mirna[data_columns_mirna].values  # Converte i dati in un array numpy
data_methy = df_methy[data_columns_methy].values  # Converte i dati in un array numpy
data_exp = df_exp[data_columns_exp].values  # Converte i dati in un array numpy
# labels = df_mirna[label_column].values  # Converte le etichette in un array numpy

# Calcolo delle matrici di affinità
affinity_networks_mirna = snf.make_affinity(data_mirna, metric='euclidean', K=20, mu=0.5, normalize=True)
print("Mirna data affinity networks created")
affinity_networks_methy = snf.make_affinity(data_methy, metric='euclidean', K=20, mu=0.5, normalize=True)
print("Methy data affinity networks created")
affinity_networks_exp = snf.make_affinity(data_exp, metric='euclidean', K=20, mu=0.5, normalize=True)
print("Exp data affinity networks created")

directory = f'{cancer_type}'
if not os.path.exists(directory):
    os.makedirs(directory)

# Salva le matrici di affinità in npy e png
np.save(f'{cancer_type}/fused_affinity_matrix_mirna.npy', affinity_networks_mirna)
np.save(f'{cancer_type}/fused_affinity_matrix_methy.npy', affinity_networks_methy)
np.save(f'{cancer_type}/fused_affinity_matrix_exp.npy', affinity_networks_exp)

# plt.imshow(affinity_networks_mirna, cmap='viridis', interpolation='nearest')
# plt.colorbar()
# plt.title('Affinity Matrix - mirna')
# plt.savefig(f'{cancer_type}/mirna_affinity_matrix_plot.png')  # Salva il plot come immagine

# plt.imshow(affinity_networks_methy, cmap='viridis', interpolation='nearest')
# plt.colorbar()
# plt.title('Affinity Matrix - methy')
# plt.savefig(f'{cancer_type}/methy_affinity_matrix_plot.png')  # Salva il plot come immagine

# plt.imshow(affinity_networks_exp, cmap='viridis', interpolation='nearest')
# plt.colorbar()
# plt.title('Affinity Matrix - exp')
# plt.savefig(f'{cancer_type}/exp_affinity_matrix_plot.png')  # Salva il plot come immagine

# Calcolo della matrice di affinità fuse
fused_affinity = snf.snf([affinity_networks_mirna, affinity_networks_methy, affinity_networks_exp], K=20)
print(f"Fused affinity networks created for {cancer_type}")

# Salva la matrice di affinità fusa in npy e png
np.save(f'{cancer_type}/fused_affinity_matrix.npy', fused_affinity)

plt.imshow(fused_affinity, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title(f'Fused Affinity Matrix - {cancer_type}')
plt.savefig(f'{cancer_type}/fused_affinity_matrix_plot.png')  # Salva il plot come immagine
'''
'''
# Determinazione del numero ottimale di cluster
best, second = snf.get_n_clusters(fused_affinity)
print(f"Best number of clusters: {best} - Second best number of clusters: {second}")

# Clustering dello SNF fused network
cluster_labels = spectral_clustering(fused_affinity, n_clusters=best)
print(cluster_labels)

# Valutazione del clustering rispetto alle etichette vere
v_measure = v_measure_score(labels, cluster_labels)
print("V-Measure Score:", v_measure)
'''

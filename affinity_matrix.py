import pandas as pd
import numpy as np
import snf
import os
# from sklearn.cluster import spectral_clustering, KMeans
# from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

cancer_type = 'SKCM1'

# Carica i dati
data_type_mirna = f"dataset/cancer/{cancer_type}_miRNA.csv"
data_type_rnaseq = f"dataset/cancer/{cancer_type}_RNASeq.csv"
data_type_rppa = f"dataset/cancer/{cancer_type}_RPPA.csv"
data_type_methy = f"dataset/cancer/{cancer_type}_Methy.csv"
df_mirna = pd.read_csv(data_type_mirna)
df_rnaseq = pd.read_csv(data_type_rnaseq)
df_rppa = pd.read_csv(data_type_rppa)
df_methy = pd.read_csv(data_type_methy, usecols=range(0, 10000)) # ci mette molto tempo a caricare il file intero

print(df_mirna.shape, df_rnaseq.shape, df_rppa.shape, df_methy.shape)

def calculate_n_components(df, proportion=0.1):
    """
    Calcola il numero di componenti da mantenere, tenendo conto che n_components non deve superare il numero di righe.
    
    :param df: Il dataframe con i dati.
    :param proportion: La proporzione delle colonne da mantenere.
    :return: Il numero di componenti da usare nell'RSVD.
    """
    n_components = int(df.shape[1] * proportion)  # Calcola il numero di componenti basato sulle colonne
    return min(n_components, df.shape[0])  # Limita il numero di componenti al numero di righe

def apply_rsvd(data_matrix, n_components):
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

n_components_mirna = calculate_n_components(df_mirna, proportion=0.2)  # 20% per miRNA
n_components_rnaseq = calculate_n_components(df_rnaseq, proportion=0.05)  # 5% per RNASeq
n_components_rppa = calculate_n_components(df_rppa, proportion=0.2)  # 20% per RPPA
n_components_methy = calculate_n_components(df_methy, proportion=0.05)  # 5% per Methy
print(n_components_mirna, n_components_rnaseq, n_components_rppa)

# Applichiamo RSVD su ogni tipo di dato
df_mirna_reduced = apply_rsvd(df_mirna, n_components_mirna)
df_rnaseq_reduced = apply_rsvd(df_rnaseq, n_components_rnaseq)
df_rppa_reduced = apply_rsvd(df_rppa, n_components_rppa)
df_methy_reduced = apply_rsvd(df_methy, n_components_methy)

print(df_mirna_reduced.shape, df_rnaseq_reduced.shape, df_rppa_reduced.shape, df_methy_reduced.shape)

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
'''

# Calcolo delle matrici di affinità
affinity_networks_mirna = snf.make_affinity(df_mirna_reduced, metric='euclidean', K=20, mu=0.5, normalize=True)
print("Mirna data affinity networks created")
affinity_networks_rnaseq = snf.make_affinity(df_rnaseq_reduced, metric='euclidean', K=20, mu=0.5, normalize=True)
print("Rnaseq data affinity networks created")
affinity_networks_rppa = snf.make_affinity(df_rppa_reduced, metric='euclidean', K=20, mu=0.5, normalize=True)
print("Rppa data affinity networks created")
affinity_networks_methy = snf.make_affinity(df_methy_reduced, metric='euclidean', K=20, mu=0.5, normalize=True)
print("Methy data affinity networks created")

directory = f'{cancer_type}'
if not os.path.exists(directory):
    os.makedirs(directory)

# Salva le matrici di affinità in npy e png
np.save(f'{cancer_type}/fused_affinity_matrix_mirna.npy', affinity_networks_mirna)
np.save(f'{cancer_type}/fused_affinity_matrix_rnaseq.npy', affinity_networks_rnaseq)
np.save(f'{cancer_type}/fused_affinity_matrix_rppa.npy', affinity_networks_rppa)
np.save(f'{cancer_type}/fused_affinity_matrix_methy.npy', affinity_networks_methy)

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
fused_affinity = snf.snf([affinity_networks_mirna, affinity_networks_rnaseq, affinity_networks_rppa, affinity_networks_methy], K=20)
print(f"Fused affinity networks created for {cancer_type}")

# Salva la matrice di affinità fusa in npy e png
np.save(f'{cancer_type}/fused_affinity_matrix.npy', fused_affinity)

plt.imshow(fused_affinity, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title(f'Fused Affinity Matrix - {cancer_type}')
plt.savefig(f'{cancer_type}/fused_affinity_matrix_plot.png')  # Salva il plot come immagine

'''
# Carica la matrice di affinità fusa e la stampa 
loaded_matrix = np.load(f'{cancer_type}/fused_affinity_matrix.npy')
print(loaded_matrix.shape)
print(loaded_matrix[:5])  # Controlla i primi 5 valori
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

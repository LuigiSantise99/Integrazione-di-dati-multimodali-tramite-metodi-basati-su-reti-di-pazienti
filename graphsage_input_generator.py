# python version 3.5.6
import networkx as nx
import numpy as np
import pandas as pd
import json
import os
import csv
import copy
from networkx.readwrite import json_graph
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

tcga_project = 'BRCA'
os.makedirs('graphsage_input/{}'.format(tcga_project), exist_ok=True)
log_dir = 'graphsage_input/{0}/{0}'.format(tcga_project)

# Carica i dati delle omiche da utilizzare
DNAm_data = pd.read_csv('../MOGDx/data/TCGA/{}/raw/datMeta_DNAm.csv'.format(tcga_project))
miRNA_data = pd.read_csv('../MOGDx/data/TCGA/{}/raw/datMeta_miRNA.csv'.format(tcga_project))
mRNA_data = pd.read_csv('../MOGDx/data/TCGA/{}/raw/datMeta_mRNA.csv'.format(tcga_project))

# Filtra solo i pazienti che compaiono in tutte e tre le omiche e unisci in un unico file senza ripetizioni
common_patients = set(DNAm_data['patient']).intersection(miRNA_data['patient']).intersection(mRNA_data['patient'])
DNAm_data = DNAm_data[DNAm_data['patient'].isin(common_patients)]
miRNA_data = miRNA_data[miRNA_data['patient'].isin(common_patients)]
mRNA_data = mRNA_data[mRNA_data['patient'].isin(common_patients)]

merged_data = pd.concat([DNAm_data, miRNA_data, mRNA_data], sort=False).drop_duplicates(subset='patient')

# Seleziona le colonne desiderate e gestisce i dati mancanti
categorical_columns = ["race", "gender", "ethnicity", "age_at_diagnosis"]
label_column = "paper_BRCA_Subtype_PAM50"
selected_columns = ["patient"] + categorical_columns + [label_column]
merged_data = merged_data[selected_columns].replace("not reported", "nan")

merged_data[categorical_columns] = merged_data[categorical_columns].astype(str)

# Codifica le colonne categoriche in formato one-hot
imputer = SimpleImputer(strategy='constant', fill_value='nan')
encoded_data = imputer.fit_transform(merged_data[categorical_columns])

# Stampa il numero di categorie e le categorie per ogni colonna (test: capire quali sono le categorie uniche per ogni colonna prima di applicare l'encoder)
# for col in categorical_columns:
#     unique_values = np.unique(encoded_data[:, categorical_columns.index(col)])
#     print("Colonna '{0}' ha {1} categorie: {2}". format(col, len(unique_values), unique_values))

encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(encoded_data)

# Codifica le etichette in formato one-hot
labels = merged_data[label_column].astype(str)
label_encoder = OneHotEncoder(sparse=False)
labels_one_hot = label_encoder.fit_transform(labels.values.reshape(-1, 1))

# Stampa il numero di categorie e le categorie per le label (test: capire quali sono le label uniche)
# label_unique_values = np.unique(labels)
# print("Colonna '{0}' ha {1} categorie: {2}".format(label_column, len(label_unique_values), label_unique_values))

# Crea la mappa degli ID dei nodi
node_ids = merged_data["patient"].tolist()
id_map = {node_id: i for i, node_id in enumerate(node_ids)}

# Crea la mappa delle classi dei nodi
class_map = {node_id: label.tolist() for node_id, label in zip(node_ids, labels_one_hot)}

# Salva i file richiesti (mappa degli ID, mappa delle classi e features)
with open('{}-id_map.json'.format(log_dir), 'w') as f:
    json.dump(id_map, f)
print('ID mapping successfully saved.')
with open('{}-class_map.json'.format(log_dir), 'w') as f:
    json.dump(class_map, f)
print('Class mapping successfully saved.')
np.save('{}-feats.npy'.format(log_dir), encoded_features)
print('Features of patients successfully saved.')

# Salva merged_data come CSV
# merged_data.to_csv('merged_data.csv', index=False)

# Salva un pddata come CSV
# labels_one_hot_df = pd.DataFrame(labels_one_hot)
# labels_one_hot_df.to_csv('labels_one_hot_df.csv', index=False)

# Carica la matrice di affinità fusa
affinity_matrix = np.load('affinity_matrices/{}/fused_affinity_matrix.npy'.format(tcga_project))

def create_graph(affinity_matrix, node_data, feats_data, patient_ids, feature_names):
    G = nx.Graph()

    # Se i nodi sono rappresentati come un array numpy, convertili in un DataFrame
    if isinstance(feats_data, np.ndarray):
        feats_data_df = pd.DataFrame(feats_data, index=patient_ids, columns=feature_names)

    # Aggiungi i nodi al grafo
    for node_id, label in node_data.items():
        features = feats_data_df.loc[node_id].tolist()
        G.add_node(node_id, label=label, features=features, val=False, test=False)
        
    # Stampa i nodi e le loro etichette (test)
#     print("Nodi e le loro etichette dopo l'aggiunta al grafo:")
#     for node_id, label in node_data.items():
#         print("Node ID: {0}, Label: {1}".format(node_id, label))
            
    # Aggiungi gli archi al grafo
    num_nodes = affinity_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if affinity_matrix[i, j] > 0:
                G.add_edge(node_ids[i], node_ids[j], train_removed=False, test_removed=False)

    # Suddivide i nodi in set di addestramento, validazione e test
    one_hot_labels = [data['label'] for _, data in G.nodes(data=True)] # array delle etichette dei nodi in formato one-hot
    labels = np.argmax(one_hot_labels, axis=1) # array delle etichette dei nodi in formato intero

    random_state=42

    train_nodes, test_nodes = train_test_split(list(G.nodes()), test_size=0.2, shuffle=True, stratify=labels, random_state=random_state) # 10% test set del set completo (20% per fare uguale a mogdx)
    for node in test_nodes:
        G.node[node]['test'] = True

    train_labels = np.argmax([G.node[node].get('label') for node in train_nodes], axis=1) # array delle etichette dei nodi di train in formato intero

    train_nodes, val_nodes = train_test_split(train_nodes, test_size=0.15, shuffle=True, stratify=train_labels, random_state=random_state) # 22,5% val set del set completo, train set 67,5% del set completo (val set 12% e train set 68% come mogdx)
    for node in val_nodes:
        G.node[node]['val'] = True
        
    # Salva file dei nodi e delle loro etichette (test per capire se le label corrispondono)
#     with open('nodi_etichette.txt', 'w') as f:
#         f.write("Nodi e le loro etichette dopo l'aggiunta al grafo:\n")
#         for node_id, data in G.nodes(data=True):
#             f.write("Node ID: {0}, Label: {1}, Val: {2}, Test: {3}\n".format(node_id, data['label'], data['val'], data['test']))
    
    return G
                
# Salva il grafo (G) come JSON
G = create_graph(affinity_matrix, class_map, encoded_features, node_ids, encoder.get_feature_names(categorical_columns))   
with open('{}-G.json'.format(log_dir), 'w') as f:
    json.dump(json_graph.node_link_data(G), f)
print('Graph successfully saved.')

# Stampa alcune informazioni sui nodi per verificare la creazione corretta (test: per verificare che label e features siano assegnate in modo corretto nel grafo)
# print("Informazioni sui nodi:")
# for i, (node_id, data) in enumerate(G.nodes(data=True)):
#     if i >= 5:  # Stampa solo i primi 10 nodi per esempio
#         break
#     print("Node ID: {0}, Features: {1}, Label: {2}".format(node_id, data.get('features'), data.get('label')))
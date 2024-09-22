'''
File di Input Richiesti da GraphSAGE:

<train_prefix>-G.json: Descrizione del grafo in formato JSON.
<train_prefix>-id_map.json: Mappa dei nodi del grafo agli ID consecutivi.
<train_prefix>-class_map.json: Mappa dei nodi del grafo alle classi.
<train_prefix>-feats.npy: Matrice delle feature dei nodi (opzionale).
<train_prefix>-walks.txt: Specifica delle co-occorrenze dei random walk (solo se il modello è n2v(?)).
'''
# python version 3.5.6
import networkx as nx
import numpy as np
import pandas as pd
import json
import os
import csv
from networkx.readwrite import json_graph
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

cancer_type = 'BLCA2'
os.makedirs('inputFilesGraphsage/{}'.format(cancer_type), exist_ok=True)
log_dir = 'inputFilesGraphsage/{0}/{0}'.format(cancer_type)

def create_graph(affinity_matrix, node_data, feats_data):
    G = nx.Graph()

    # Aggiungi i nodi al grafo
    for node_id, death in node_data.items():
        label = [1, 0] if death == 0 else [0, 1] # codifica one-hot
        features = feats_data[node_id].tolist()
        G.add_node(node_id, label=label, features=features, val=False, test=False)

    # Aggiungi gli archi al grafo
    num_nodes = affinity_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if affinity_matrix[i, j] > 0:
                 G.add_edge(i, j, train_removed=False, test_removed=False)

    # Dividi i nodi in training e validation set
    train_nodes, val_nodes = train_test_split(list(G.nodes()), test_size=0.2, random_state=42)
    for node in val_nodes:
        G.node[node]['val'] = True

    # Dividi i nodi in training e test set
    train_nodes, test_nodes = train_test_split(train_nodes, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    for node in test_nodes:
        G.node[node]['test'] = True

    return G

def select_features(feats_data):

    # Seleziona le colonne categoriche
    categorical_column = ["years_to_birth","race","gender","ethnicity","patient.age_at_initial_pathologic_diagnosis"]

    categorical_data = feats_data[categorical_column]
    categorical_data = categorical_data.astype(str)
    print(categorical_data)

    # Initialize SimpleImputer to fill NaN values with a placeholder
    imputer = SimpleImputer(strategy='constant', fill_value='missing')
    # Apply SimpleImputer
    imputed_categorical_data = imputer.fit_transform(categorical_data)
    # Inizializza OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    # Applica OneHotEncoder e trasforma le colonne categoriche
    encoded_features = encoder.fit_transform(imputed_categorical_data)

    # Creazione del DataFrame delle feature codificate
    encoded_feats_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names(categorical_column))
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    print(encoded_feats_data)

    return encoded_feats_data

# Carica la matrice di affinità fusa
affinity_matrix = np.load('{}/fused_affinity_matrix.npy'.format(cancer_type))

# Salva le classi dei nodi
labels = []
node_ids = []
with open("dataset/label/{}_os.csv".format(cancer_type), 'r', newline='') as csv_file:
    reader = csv.DictReader(csv_file)
    temp_map = {}

    for row in reader:
        node_id = row[''] # ID del nodo
        death = int(row['nn'])  # Classe del nodo
        temp_map[node_id] = death
        node_ids.append(node_id)
    
    for node_id in node_ids:
        labels.append(temp_map[node_id])

class_map = {node_id: label for node_id, label in zip(node_ids, labels)}

with open('{}-class_map.json'.format(log_dir), 'w') as f:
    json.dump(class_map, f)
print('Classi dei nodi salvate come JSON')

# Salva la mappa degli ID dei nodi
id_map = {node_id: i for i, node_id in enumerate(node_ids)}
with open('{}-id_map.json'.format(log_dir), 'w') as f:
    json.dump(id_map, f)
print('Mappa degli ID dei nodi salvata come JSON')


# Salva le features dei nodi
feats_data = pd.read_csv("dataset/clinical/{}_clinics.csv".format(cancer_type), sep=',')
features = select_features(feats_data).to_numpy()
np.save('{}-feats.npy'.format(log_dir), features)
print('Matrice delle feature dei nodi salvata come .npy')
#print(features)


# Salva il grafo (G) come JSON
G = create_graph(affinity_matrix, class_map, features)
with open('{}-G.json'.format(log_dir), 'w') as f:
    json.dump(json_graph.node_link_data(G), f)
print('Grafo salvato come JSON')









'''
colonne_interesse = ['age_at_initial_pathologic_diagnosis']
df_selezionato = feats_data[colonne_interesse]
print(df_selezionato)
'''

'''
# Specifica il percorso del file .npy
file_path = '../GraphSAGE/example_data/toy-ppi-feats.npy'

# Carica il file .npy
data = np.load(file_path)

# Stampa il contenuto
df = pd.DataFrame(data)

# Stampa il DataFrame
print(df)
'''
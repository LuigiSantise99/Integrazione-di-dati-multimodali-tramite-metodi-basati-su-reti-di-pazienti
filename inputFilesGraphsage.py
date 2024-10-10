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

cancer_type = 'PRAD1'
os.makedirs('inputFilesGraphsage/{}'.format(cancer_type), exist_ok=True)
log_dir = 'inputFilesGraphsage/{0}/{0}'.format(cancer_type)

def create_graph(affinity_matrix, node_data, feats_data, patient_ids, feature_names):
    G = nx.Graph()

    # Se i nodi sono rappresentati come un array numpy, convertili in un DataFrame
    if isinstance(feats_data, np.ndarray):
        feats_data_df = pd.DataFrame(feats_data, index=patient_ids, columns=feature_names)
    
    feats_data_df.to_csv(os.path.join('inputFilesGraphsage/', 'feats_data_df.csv'), index=True)

    # Aggiungi i nodi al grafo
    for node_id, death in node_data.items():
        label = [1, 0] if death == 0 else [0, 1] # codifica one-hot
        features = feats_data_df.loc[node_id].tolist()
        G.add_node(node_id, label=label, features=features, val=False, test=False)
        
            
    # Aggiungi gli archi al grafo
    num_nodes = affinity_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if affinity_matrix[i, j] > 0:
                 G.add_edge(node_ids[i], node_ids[j], train_removed=False, test_removed=False)

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
    categorical_columns = ["patientID", "years_to_birth", "gender", "ethnicity", 
                          "patient.age_at_initial_pathologic_diagnosis"]
    
    # Verifica quale colonna "race" è presente con priorità a "race"
    if 'race' in feats_data.columns:
        categorical_columns.append('race')
    elif 'patient.race' in feats_data.columns:
        categorical_columns.append('patient.race')
    elif 'patient.clinical_cqcf.race' in feats_data.columns:
        categorical_columns.append('patient.clinical_cqcf.race')

    # Trova le colonne presenti sia nel DataFrame che nella lista categorical_columns
    available_columns = list(set(categorical_columns).intersection(feats_data.columns))

    categorical_data = feats_data[available_columns]
    print(categorical_data.head())
    categorical_data.set_index('patientID', inplace=True)
    categorical_data = categorical_data.astype(str)
    #print(categorical_data)

    # Initialize SimpleImputer to fill NaN values with a placeholder
    imputer = SimpleImputer(strategy='constant', fill_value='missing')
    # Apply SimpleImputer
    imputed_categorical_data = imputer.fit_transform(categorical_data)
    # Inizializza OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    # Applica OneHotEncoder e trasforma le colonne categoriche
    encoded_features = encoder.fit_transform(imputed_categorical_data)

    # Creazione del DataFrame delle feature codificate
    encoded_feats_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names(categorical_data.columns))
    encoded_feats_data.index = categorical_data.index
    
    return encoded_feats_data.to_numpy(), encoder.get_feature_names(categorical_data.columns), categorical_data.index

# Carica la matrice di affinità fusa
affinity_matrix = np.load('{}/fused_affinity_matrix.npy'.format(cancer_type))

# Salva la mappa degli ID dei nodi e le classi dei nodi
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

id_map = {node_id: i for i, node_id in enumerate(node_ids)}
with open('{}-id_map.json'.format(log_dir), 'w') as f:
    json.dump(id_map, f)
print('Mappa degli ID dei nodi salvata come JSON')

class_map = {node_id: label for node_id, label in zip(node_ids, labels)}
with open('{}-class_map.json'.format(log_dir), 'w') as f:
    json.dump(class_map, f)
print('Classi dei nodi salvate come JSON')


# Salva le features dei nodi
feats_data = pd.read_csv("dataset/clinical/{}_clinics.csv".format(cancer_type), sep=',')
encoded_features, feature_names, patient_ids = select_features(feats_data)
np.save('{}-feats.npy'.format(log_dir), encoded_features)
np.savetxt('inputFilesGraphsage/feats_data.txt', encoded_features, delimiter=',')
print('Matrice delle feature dei nodi salvata come .npy')

# Salva il grafo (G) come JSON
G = create_graph(affinity_matrix, class_map, encoded_features, patient_ids, feature_names)
with open('{}-G.json'.format(log_dir), 'w') as f:
    json.dump(json_graph.node_link_data(G), f)
print('Grafo salvato come JSON')

'''
def check_features_for_id(node_id):
    if node_id in id_map:
        # Ottieni l'indice associato all'ID specifico dall'id_map
        index = id_map[node_id]

        # Ottieni le features salvate in feats.npy per quell'indice
        saved_features = encoded_features[index]

        # Trova il nodo nel grafo con lo stesso ID
        node_in_graph = next((data for node, data in G.nodes(data=True) if node == node_id), None)

        if node_in_graph:
            graph_features = node_in_graph['features']

            # Confronta le features
            if np.allclose(graph_features, saved_features):
                print("Le features per il nodo {} corrispondono.".format(node_id))
                print("Features nel grafo: {}".format(graph_features))
                print("Features salvate in feats.npy: {}".format(saved_features))
            else:
                print("Le features per il nodo {} NON corrispondono.".format(node_id))
                print("Features nel grafo: {}".format(graph_features))
                print(len(graph_features))
                print("Features salvate in feats.npy: {}".format(saved_features))
                print(len(saved_features))

            # Trova le posizioni degli 1 nelle features salvate e in quelle del grafo
            saved_positions = [i for i, x in enumerate(saved_features) if x == 1]
            graph_positions = [i for i, x in enumerate(graph_features) if x == 1]

            # Confronta le posizioni degli 1
            if saved_positions == graph_positions:
                print("Le posizioni degli 1 per il nodo {} corrispondono.".format(node_id))
                print("Posizioni degli 1 nel grafo: {}".format(graph_positions))
                print("Posizioni degli 1 salvate in feats.npy: {}".format(saved_positions))
            else:
                print("Le posizioni degli 1 per il nodo {} NON corrispondono.".format(node_id))
                print("Posizioni degli 1 nel grafo: {}".format(graph_positions))
                print("Posizioni degli 1 salvate in feats.npy: {}".format(saved_positions))
        else:
            print("Il nodo con ID {} non è stato trovato nel grafo.".format(node_id))
    else:
        print("L'ID {} non è presente in id_map.".format(node_id))

# Specifica l'ID del nodo che vuoi controllare
node_id_to_check = "TCGA-BL-A13I"
check_features_for_id(node_id_to_check)
'''

def check_all_features():
    all_match = True

    for node_id in node_ids:
        if node_id in id_map:
            index = id_map[node_id]
            saved_features = encoded_features[index]
            node_in_graph = next((data for node, data in G.nodes(data=True) if node == node_id), None)

            if node_in_graph:
                graph_features = node_in_graph['features']
                if not np.allclose(graph_features, saved_features):
                    all_match = False
                    break
            else:
                all_match = False
                break
        else:
            all_match = False
            break

    if all_match:
        print("Tutte le features combaciano.")
    else:
        print("Almeno una feature non combacia.")

check_all_features()



'''
# Carica il file .npy
npy_file_path = '../GraphSAGE/example_data/toy-ppi-feats.npy'
data = np.load(npy_file_path)

# Salva il contenuto in un file .txt
txt_file_path = 'inputFilesGraphsage/feats_data_example.txt'
np.savetxt(txt_file_path, data, delimiter=',')
print('File .npy convertito e salvato come .txt')
'''

'''
if G is not None:
    with open('{}-G.json'.format(log_dir), 'w') as f:
        json.dump(json_graph.node_link_data(G), f)
    print('Grafo salvato come JSON')

    # Estrai gli ID dal grafo
    graph_ids = [node for node in G.nodes()]
    
    # Controlla che ogni ID del grafo sia in id_map
    missing_ids = [id_ for id_ in graph_ids if id_ not in id_map]
    if missing_ids:
        print("Questi ID del grafo non sono presenti in id_map:", missing_ids)
    else:
        print("Tutti gli ID del grafo sono presenti in id_map.")
else:
    print("Failed to create graph due to input errors.")

'''



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
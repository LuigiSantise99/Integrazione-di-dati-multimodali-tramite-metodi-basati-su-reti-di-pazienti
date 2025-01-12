# python version 3.5.6
import networkx as nx
import numpy as np
import pandas as pd
import json
import os
import csv
from collections import Counter, OrderedDict
from networkx.readwrite import json_graph
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

tcga_project = 'BRCA'
os.makedirs('graphsage_input/{}'.format(tcga_project), exist_ok=True)
log_dir = 'graphsage_input/{0}/{0}'.format(tcga_project)

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
        
    # Salva file dei nodi e delle loro etichette (test: per capire se le label corrispondono)
#     with open('nodi_etichette.txt', 'w') as f:
#         f.write("Nodi e le loro etichette dopo l'aggiunta al grafo:\n")
#         for node_id, data in G.nodes(data=True):
#             f.write("Node ID: {0}, Label: {1}, Val: {2}, Test: {3}\n".format(node_id, data['label'], data['val'], data['test']))
    
    return G

def create_graph_from_csv(g_csv, node_data, feats_data, patient_ids, feature_names):    
    # Crea il grafo
    G = nx.Graph()
    
    # Aggiungi i nodi con etichette e caratteristiche
    if isinstance(feats_data, np.ndarray):
        feats_data_df = pd.DataFrame(feats_data, index=patient_ids, columns=feature_names)

    for node_id, label in node_data.items():
        features = feats_data_df.loc[node_id].tolist()
        G.add_node(node_id, label=label, features=features, val=False, test=False)
        
    # Aggiungi gli archi dal CSV
    for _, row in g_csv.iterrows():
        G.add_edge(row['from_name'], row['to_name'], train_removed=False, test_removed=False)
    
    # Suddividi i nodi in train, validation e test
    one_hot_labels = [data['label'] for _, data in G.nodes(data=True)]
    labels = np.argmax(one_hot_labels, axis=1)
    
    random_state = 42
    train_nodes, test_nodes = train_test_split(list(G.nodes()), test_size=0.2, shuffle=True, stratify=labels, random_state=random_state)
    for node in test_nodes:
        G.node[node]['test'] = True
    
    train_labels = np.argmax([G.node[node].get('label') for node in train_nodes], axis=1)

    train_nodes, val_nodes = train_test_split(train_nodes, test_size=0.15, shuffle=True, stratify=train_labels, random_state=random_state)
    for node in val_nodes:
        G.node[node]['val'] = True

    # Conteggio nodi in ciascun set e cardinalità delle label (test: quanti nodi per ogni set e quanti nodi per ogni label in ogni set)
#     label_mapping = {0: "Basal", 1: "Her2", 2: "LumA", 3: "LumB", 4: "Normal"}
#     ordered_labels = ["Basal", "Her2", "LumA", "LumB", "Normal"]
#     train_nodes = [node for node in G.nodes() if not G.node[node].get('test') and not G.node[node].get('val')]
#     val_nodes = [node for node in G.nodes() if G.node[node].get('val')]
#     test_nodes = [node for node in G.nodes() if G.node[node].get('test')]
#     print("Numero di nodi in training set: {}".format(len(train_nodes)))
#     print("Numero di nodi in validation set: {}".format(len(val_nodes)))
#     print("Numero di nodi in test set: {}".format(len(test_nodes)))
    
#     train_label_counts = OrderedDict((label, Counter([label_mapping[np.argmax(G.node[node]['label'])] for node in train_nodes]).get(label, 0)) for label in ordered_labels)
#     val_label_counts = OrderedDict((label, Counter([label_mapping[np.argmax(G.node[node]['label'])] for node in val_nodes]).get(label, 0)) for label in ordered_labels)
#     test_label_counts = OrderedDict((label, Counter([label_mapping[np.argmax(G.node[node]['label'])] for node in test_nodes]).get(label, 0)) for label in ordered_labels)

#     print("Cardinalità delle label:")
#     print("Training set: {}".format(train_label_counts))
#     print("Validation set: {}".format(val_label_counts))
#     print("Test set: {}".format(test_label_counts))
    
    return G

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
merged_data = merged_data.sort_values('patient')
merged_data = merged_data.reset_index(drop=True)

# Raggruppa age_at_diagnosis per decadi
merged_data['age_at_diagnosis'] = pd.to_numeric(merged_data['age_at_diagnosis'], errors='coerce')
merged_data['age_at_diagnosis_years'] = merged_data['age_at_diagnosis'] / 365
merged_data['age_at_diagnosis_decade'] = merged_data['age_at_diagnosis_years'].apply(
    lambda x: "{}".format(int(x // 10 * 10)) if pd.notnull(x) else np.nan) 

# Seleziona le colonne desiderate e gestisce i dati mancanti
categorical_columns = ["race", "ethnicity", "age_at_diagnosis_decade"] #non uso gender per BRCA
label_column = "paper_BRCA_Subtype_PAM50"
selected_columns = ["patient"] + categorical_columns + [label_column]
merged_data = merged_data[selected_columns].replace("not reported", np.nan)

#creazione dei grafi (test: distribuzione delle categorie per ogni label) *per funziona sostituire np.nan nella stringa 'np.nan'
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# save_as_pdf = True
# output_file = "feature_distributions.pdf" if save_as_pdf else "feature_distributions.png"

# num_features = len(categorical_columns)
# cols = 1
# rows = (num_features + cols - 1) // cols

# fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 8), constrained_layout=True)
# axes = axes.flatten()

# for idx, column in enumerate(categorical_columns):
#     ax = axes[idx]
#     grouped_data = merged_data.groupby(label_column)
#     categories = merged_data[column].fillna("np.nan").unique()
    
#     data = {cat: [group[group[column].fillna("np.nan") == cat].shape[0] for label, group in grouped_data]
#             for cat in categories}

#     y = np.arange(len(grouped_data))
#     height = 0.15

#     colors = plt.cm.get_cmap("tab10", len(categories))

#     if column == "age_at_diagnosis_decade":
#         height = 0.1
#         ax.set_title("Distribuzione per 'age_at_diagnosis_decade'")
#         ax.set_yticks(y + (len(categories) - 1) * height / 2)
#         ax.set_yticklabels(['20', '30', '40', '50', '60', '70', '80', '90', 'np.nan'])
#         ax.set_ylim(-0.2, len(grouped_data))
#     else:
#         ax.set_title("Distribuzione delle categorie per '{}'".format(column))

#     for i, (category, counts) in enumerate(data.items()):
#         ax.barh(y + i * height, counts, height, label=category, color=colors(i))

#     ax.set_ylabel("Classi")
#     ax.set_xlabel("Conteggio")
#     ax.set_yticks(y + (len(categories) - 1) * height / 2)
#     ax.set_yticklabels(['Normal', 'LumB', 'LumA', 'Her2', 'Basal'])
#     ax.legend(title="Categorie", bbox_to_anchor=(1.05, 1), loc="upper left")

# for idx in range(num_features, len(axes)):
#     fig.delaxes(axes[idx])

# if save_as_pdf:
#     with PdfPages(output_file) as pdf:
#         pdf.savefig(fig)
# else:
#     plt.savefig(output_file, dpi=300)

# print("Grafici salvati in: {}".format(output_file))
#fine test

# Conteggio delle occorrenze della stringa "nan" per ogni colonna (test: per capire quanti valori nulli in ogni categoria)
# nan_string_counts = (merged_data[categorical_columns] == 'nan').sum()
# print("Conteggio delle occorrenze della stringa 'nan' per ogni colonna:")
# print(nan_string_counts)

#  Codifica le colonne categoriche in formato one-hot
imputer = SimpleImputer(strategy='most_frequent')
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
# # Conta il numero di pazienti per ogni label (test: per capire il totale di pazienti per ogni etichetta)
# print("Totale di pazienti per ogni label:")
# print(merged_data[label_column].value_counts())

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

# Salva il file delle features come dataframe id_paziente x features
df_encoded_features = pd.DataFrame(encoded_features, index=node_ids)
df_encoded_features.to_csv('{}-features_df.csv'.format(log_dir))

# Carica la matrice di affinità fusa
affinity_matrix = np.load('affinity_matrices/{}/fused_affinity_matrix.npy'.format(tcga_project))
                
# Salva il grafo (G) come JSON
# G = create_graph(affinity_matrix, class_map, encoded_features, node_ids, encoder.get_feature_names(categorical_columns))   
# with open('{}-G.json'.format(log_dir), 'w') as f:
#     json.dump(json_graph.node_link_data(G), f)
# print('Graph successfully saved.')

# Stampa alcune informazioni sui nodi per verificare la creazione corretta (test: per verificare che label e features siano assegnate in modo corretto nel grafo)
# print("Informazioni sui nodi:")
# for i, (node_id, data) in enumerate(G.nodes(data=True)):
#     if i >= 5:  # Stampa solo i primi 10 nodi per esempio
#         break
#     print("Node ID: {0}, Features: {1}, Label: {2}".format(node_id, data.get('features'), data.get('label')))

# Converte il grafo csv in un grafo json completo di caratteristiche e label e split set
g_csv = pd.read_csv('{}-G.csv'.format(log_dir)) #se uso snfpy
# g_csv = pd.read_csv('graphsage_input/BRCA/mRNA_miRNA_DNAm_graph.csv'.format(log_dir)) #se affinity_matrix creata seguendo pipeline di mogdx (nomde DA CAMBIARE DURANTE SALVATAGGIO)
G = create_graph_from_csv(g_csv, class_map, encoded_features, node_ids, encoder.get_feature_names(categorical_columns))

with open('{}-G.json'.format(log_dir), 'w') as f:
    json.dump(json_graph.node_link_data(G), f)
print('Graph successfully saved.')

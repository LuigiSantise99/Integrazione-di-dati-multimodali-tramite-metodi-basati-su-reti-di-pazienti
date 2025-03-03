# python version 3.5.6
import networkx as nx
import numpy as np
import pandas as pd
import json
import os
import csv
from collections import Counter, OrderedDict
from networkx.readwrite import json_graph
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

tcga_project = 'BRCA'
os.makedirs('graphsage_input/{}'.format(tcga_project), exist_ok=True)
log_dir = 'graphsage_input/{0}/{0}'.format(tcga_project)
exp_n = 'exp_3' #tipo di esperimento

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
    
    # Suddivisione in train, val e test set (k-fold cv)
    # Ottieni le label dei nodi
    one_hot_labels = [data['label'] for _, data in G.nodes(data=True)]
    labels = np.argmax(one_hot_labels, axis=1)
    nodes = list(G.nodes())

    # Inizializza lo StratifiedKFold
    random_state = 42
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Per ogni fold
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(nodes, labels)):
  
        # Crea una copia del grafo per il fold
        G_fold = G.copy()
        
        # Dividi i nodi in train/val e test
        train_val_nodes = [nodes[i] for i in train_val_idx]
        test_nodes = [nodes[i] for i in test_idx]

        # Assegna i nodi di test come True
        for node in test_nodes:
            G_fold.node[node]['test'] = True
    
        # Dividi train_val in train e validation
        train_val_labels = np.array([labels[i] for i in train_val_idx])
        train_idx, val_idx = train_test_split(
            range(len(train_val_nodes)),
            test_size=0.15,
            shuffle=True,
            stratify=train_val_labels,
            random_state=random_state
        )
    
        train_nodes = [train_val_nodes[i] for i in train_idx]
        val_nodes = [train_val_nodes[i] for i in val_idx]
    
        # Assegna i nodi di validation come True
        for node in val_nodes:
            G_fold.node[node]['val'] = True

        # Salva il grafo in formato JSON
        output_file = "graphsage_input/{0}/{1}/{0}-G{2}.json".format(tcga_project, exp_n, fold_idx + 1)
        with open(output_file, 'w') as f:
            json.dump(json_graph.node_link_data(G_fold), f)
    
        print("Grafo {0}/{1} salvato.".format(fold_idx + 1, n_splits))
        print("Train nodes: {0}, Validation nodes: {1}, Test nodes: {2}".format(len(train_nodes), len(val_nodes), len(test_nodes)))


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
    lambda x: "{}".format(int(x // 10 * 10)) if pd.notnull(x) else 'null') #sostituire 'null' con np.nan

# Seleziona le colonne desiderate e gestisce i dati mancanti
categorical_columns = ["race", "ethnicity", "age_at_diagnosis_decade"] #non uso gender per BRCA
label_column = "paper_BRCA_Subtype_PAM50"
selected_columns = ["patient"] + categorical_columns + [label_column]
merged_data = merged_data[selected_columns].replace("not reported", 'null') #sostituire 'null' con np.nan

# creazione dei grafi (test: distribuzione delle categorie per ogni label) *per funziona sostituire np.nan nella stringa 'null'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

save_as_pdf = True
output_file = "feature_distributions.pdf" if save_as_pdf else "feature_distributions.png"

num_features = len(categorical_columns)
cols = 1
rows = (num_features + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 8), constrained_layout=True)
axes = axes.flatten()

for idx, column in enumerate(categorical_columns):
    ax = axes[idx]
    grouped_data = merged_data.groupby(label_column)
    categories = merged_data[column].fillna("null").unique()
    
    data = {cat: [group[group[column].fillna("null") == cat].shape[0] for label, group in grouped_data]
            for cat in categories}

    y = np.arange(len(grouped_data))
    height = 0.15

    colors = plt.cm.get_cmap("tab10", len(categories))

    if column == "age_at_diagnosis_decade":
        height = 0.1
        ax.set_title("Distribuzione per 'age_at_diagnosis_decade'")
        ax.set_yticks(y + (len(categories) - 1) * height / 2)
        ax.set_yticklabels(['20', '30', '40', '50', '60', '70', '80', '90', 'null'])
        ax.set_ylim(-0.2, len(grouped_data))
    else:
        ax.set_title("Distribuzione delle categorie per '{}'".format(column))

    for i, (category, counts) in enumerate(data.items()):
        ax.barh(y + i * height, counts, height, label=category, color=colors(i))

    ax.set_ylabel("Classi")
    ax.set_xlabel("Conteggio")
    ax.set_yticks(y + (len(categories) - 1) * height / 2)
    ax.set_yticklabels(['Normal', 'LumB', 'LumA', 'Her2', 'Basal'])
    ax.legend(title="Categorie", bbox_to_anchor=(1.05, 1), loc="upper left")

for idx in range(num_features, len(axes)):
    fig.delaxes(axes[idx])

if save_as_pdf:
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig)
else:
    plt.savefig(output_file, dpi=300)

print("Grafici salvati in: {}".format(output_file))
# fine test

# Supponiamo che `merged_data` sia il tuo DataFrame e `categorical_columns` contenga le colonne categoriche
label_c = "paper_BRCA_Subtype_PAM50"  # Sostituisci con il nome corretto della colonna delle etichette

# Creazione del DataFrame per l'output
output_data = []

for column in categorical_columns:
    grouped_data = merged_data.groupby(label_c)
    categories = merged_data[column].fillna("null").unique()
    
    for category in categories:
        row = {"Category": column, "Value": category}
        row.update({label: (group[column].fillna("null") == category).sum() for label, group in grouped_data})
        output_data.append(row)

# Creazione del DataFrame finale
df_output = pd.DataFrame(output_data)

# Salvataggio in un file Excel
output_excel = "category_counts.xlsx"
df_output.to_excel(output_excel, index=False)

print("File Excel creato: {}".format(output_excel))




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
print("Totale di pazienti per ogni label:")
print(merged_data[label_column].value_counts())

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
# df_encoded_features = pd.DataFrame(encoded_features, index=node_ids)
# df_encoded_features.to_csv('{}-features_df.csv'.format(log_dir))

# Carica la matrice di affinità fusa
affinity_matrix = np.load('affinity_matrices/{}/fused_affinity_matrix.npy'.format(tcga_project))

# Stampa alcune informazioni sui nodi per verificare la creazione corretta (test: per verificare che label e features siano assegnate in modo corretto nel grafo)
# print("Informazioni sui nodi:")
# for i, (node_id, data) in enumerate(G.nodes(data=True)):
#     if i >= 5:  # Stampa solo i primi 10 nodi per esempio
#         break
#     print("Node ID: {0}, Features: {1}, Label: {2}".format(node_id, data.get('features'), data.get('label')))

# Converte il grafo csv in un grafo json completo di caratteristiche, label e split set
g_csv = pd.read_csv('graphsage_input/{0}/{1}/{0}-G.csv'.format(tcga_project, exp_n))
create_graph_from_csv(g_csv, class_map, encoded_features, node_ids, encoder.get_feature_names(categorical_columns))

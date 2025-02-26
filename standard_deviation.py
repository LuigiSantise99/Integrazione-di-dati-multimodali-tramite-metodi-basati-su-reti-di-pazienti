import os
import re
import numpy as np

###############################################################################
# CONFIGURAZIONE DI BASE
###############################################################################

cancer_type = 'BRCA'
exp_type    = 'exp_1'      # Puoi cambiare a piacere (es. 'exp_2', 'exp_3', ...)
use_features = True        # Se True => 'with_features', altrimenti 'no_features'

# Converte la variabile boolean in una stringa (per costruire il path di output)
features_str = 'with_features' if use_features else 'no_features'

# Percorso di input: dove si trovano i grafi (in base a come hai organizzato le cartelle)
# Esempio:
#   "graphsage_input/BRCA/exp_1/exp_1-features/metriche/graph1/test_stats.txt"
# oppure
#   "graphsage_input/BRCA/exp_1/exp_1-no_features/metriche/graph1/test_stats.txt"
input_dir = "graphsage_input/{0}/{1}/{1}-features".format(cancer_type, exp_type) if use_features else \
            "graphsage_input/{0}/{1}/{1}-nofeatures".format(cancer_type, exp_type)

###############################################################################
# FUNZIONI DI UTILITÀ
###############################################################################

def parse_metrics(filepath):
    """
    Legge un file in cui potrebbero esserci coppie 'chiave=valore'
    su più righe o tutte in una riga sola, separate da spazi.
    Restituisce un dizionario {chiave: valore_float}.
    """
    metrics = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        # Leggiamo tutto il contenuto
        content = f.read().strip()
        
        # Cerchiamo tutte le sottostringhe del tipo chiave=valore,
        # es. 'loss=0.57899', 'f1_micro=0.80576', ecc.
        pairs = re.findall(r'(\S+)=(\S+)', content)
        for key, val_str in pairs:
            # Proviamo a convertire val_str in float
            try:
                val = float(val_str)
            except ValueError:
                # se non si converte, lasciamo la stringa (ma di solito non succede)
                val = val_str
            metrics[key] = val

    return metrics

def aggregate_metrics_across_graphs(file_name="test_stats.txt", num_graphs=5):
    """
    Legge il file `file_name` da graph1, graph2, ..., graphN (num_graphs),
    estrae le metriche e le aggrega in un dizionario:
      {
         'accuracy': [val_grafo1, val_grafo2, ..., val_grafoN],
         'precision': [...],
         ...
      }
    """
    all_metrics = {}  # Dizionario di liste: {metric_name: [v1, v2, v3, ...]}

    for i in range(1, num_graphs+1):
        filepath = os.path.join(input_dir, "metriche", "graph{}".format(i), file_name)
        if not os.path.exists(filepath):
            print("ATTENZIONE: file non trovato -> {}".format(filepath))
            continue

        metrics_dict = parse_metrics(filepath)
        # Per ogni metrica trovata nel file
        for m_name, m_value in metrics_dict.items():
            if m_name not in all_metrics:
                all_metrics[m_name] = []
            all_metrics[m_name].append(m_value)

    return all_metrics

def compute_mean_std(all_metrics):
    """
    Calcola media e deviazione standard per ogni metrica presente in `all_metrics`.
    all_metrics è del tipo { 'accuracy': [v1, v2, ...], 'precision': [v1, v2, ...] }
    Restituisce due dizionari: (mean_dict, std_dict).
    """
    mean_dict = {}
    std_dict = {}

    for m_name, values_list in all_metrics.items():
        arr = np.array(values_list, dtype=np.float32)
        mean_dict[m_name] = np.mean(arr)
        std_dict[m_name]  = np.std(arr)

    return mean_dict, std_dict


###############################################################################
# 1. AGGREGA LE METRICHE DEI 5 GRAFI
###############################################################################
test_stats_aggregated = aggregate_metrics_across_graphs(file_name="test_stats.txt", num_graphs=5)
val_stats_aggregated  = aggregate_metrics_across_graphs(file_name="val_stats.txt",  num_graphs=5)

###############################################################################
# 2. CALCOLA MEDIA E DEVIAZIONE STANDARD
###############################################################################
test_mean, test_std = compute_mean_std(test_stats_aggregated)
val_mean,  val_std  = compute_mean_std(val_stats_aggregated)

###############################################################################
# 3. CREAZIONE DELLA CARTELLA DI OUTPUT
###############################################################################
# Esempio: "results/exp_1/with_features/" (o "results/exp_1/no_features/")
output_dir = os.path.join("results", exp_type, features_str)
os.makedirs(output_dir, exist_ok=True)

###############################################################################
# 4. SALVATAGGIO SU FILE
###############################################################################
# Ciascun file conterra' le metriche incolonnate, ad esempio:
# Metrica, Media, DeviazioneStandard
# accuracy, 0.854, 0.012
# precision, 0.795, 0.020
# ...

def save_metrics_file(filename, mean_dict, std_dict):
    """
    Salva un file CSV-like con la struttura:
        Metrica,Media,Std
        accuracy,0.85,0.01
        precision,0.79,0.02
        ...
    """
    lines = ["Metrica,Media,Std"]
    # ordiniamo le metriche in ordine alfabetico per comodità
    for m_name in sorted(mean_dict.keys()):
        m = mean_dict[m_name]
        s = std_dict[m_name]
        line = "{0},{1:.5f},{2:.5f}".format(m_name, m, s)
        lines.append(line)

    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

# Salviamo i risultati per "test_stats" e "val_stats"
save_metrics_file("final_test_metrics.csv", test_mean, test_std)
save_metrics_file("final_val_metrics.csv", val_mean,  val_std)

print("File salvati nella cartella:", output_dir)
print("Contenuto test_mean:", test_mean)
print("Contenuto test_std :", test_std)
print("Contenuto val_mean :", val_mean)
print("Contenuto val_std  :", val_std)

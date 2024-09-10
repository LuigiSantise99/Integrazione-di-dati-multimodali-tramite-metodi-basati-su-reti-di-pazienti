import numpy as np

xs = ['gbm', 'liver', 'kidney', 'melanoma', 'sarcoma']

for x in xs:
    # Carica la matrice di affinità dal file .npy
    affinity_matrix = np.load(f'./{x}/fused_affinity_matrix.npy')

    # Ottieni il numero di variabili (campioni)
    num_variables = affinity_matrix.shape[0]

    print(f'Il numero di variabili (campioni) nella matrice di affinità per {x} è: {num_variables}')

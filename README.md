# Integrazione di dati multimodali tramite metodi basati su reti di pazienti

## Descrizione
Questo repository contiene il progetto di tesi per il corso di laurea magistrale di Informatica, svolto presso l'Università Statale di Milano. Include script Python, R, documentazione e tesi.
L’obiettivo generale di questo lavoro è stato sviluppare un metodo per l'integrazione di dati multimodali provenienti da pazienti oncologici e la loro successiva classificazione in base al sottotipo patologico.

## Struttura della Repository
- **affinity_matrix.py**: utilizza SNF per l'integrazione nel primo esperimento.
- **inputFilesGraphsage**: crea i file di input per l'addestramento del modello GraphSAGE.
- **graph_creation.R**: crea il grafo integrato per il primo esperimento.
- **standard_deviation.py**: calcolo della media e della deviazione standard per i risultati di GraphSAGE.
- **tesi**: cartella che contiene la tesi e presentazione.

## Modifiche apportate al codice
**Modifiche al modello GraphSAGE**: 
- Sono state aggiunte nuove metriche di valutazione che non erano presenti nella versione originale: f1 weighted, accuracy, AUPRC e AUC.
- Le modifiche sono state apportate alle funzioni evaluate() e incremental_evaluate() nel file supervised_train.py.
- Le metriche sono state aggiunte anche ai plot creati su TensorFlow.

**Modifiche a MOGDx**:
- Seed nella funzione cvtrait() nel file preprocess_functions.R: Aggiunto un parametro per migliorare la gestione della riproducibilità dei risultati.
- Aggiunta funzione snf.to.graphfromPy nel file preprocess_functions.R: sparsificazione per il primo esperimento.
- Modifiche in SNF.R: per garantire che il grafo CSV venga creato utilizzando solo i pazienti in comune tra le modalità selezionate.
- Modifiche in knn_graph_generation.R: apportate per il terzo esperimento, viene usata una funzione (select_by_sds() creata nel file preprocess_functions.R) per selezionare le 500 feature con la maggiore variabilità.

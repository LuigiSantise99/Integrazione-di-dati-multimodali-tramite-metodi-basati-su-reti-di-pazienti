library(reticulate)
np <- import("numpy")
source('../MOGDx/PreProcessing/R_Preprocessing/preprocess_functions.R')

setwd('~/tesi-progetto')

dataset <- 'TCGA'
project <- 'BRCA'
trait <- c('paper_BRCA_Subtype_PAM50')
index_col <- 'patient'
modalities <- c( 'mRNA' , 'miRNA' , 'DNAm' )

W <- np$load("affinity_matrices/BRCA/fused_affinity_matrix.npy")

colnames <- c('patient' ,  'race' , 'gender'  , 'ethnicity', 'age_at_diagnosis', trait)
datMeta <- t(data.frame( row.names = colnames))
for (mod in modalities) {
    print(mod)
    datMeta <- rbind(datMeta , read.csv(paste0('../MOGDx/data/',dataset,'/',project,'/raw/datMeta_',mod,'.csv') , row.names = 1)[ , colnames])
}
datMeta <- datMeta[!(duplicated(datMeta)),]
rownames(datMeta) <- datMeta[[index_col]]
print(dim(datMeta))

# Percorso di base per i file datMeta
base_path <- "../MOGDx/data/TCGA/BRCA/raw/"

# File datMeta da caricare
files <- c("datMeta_mRNA.csv", "datMeta_miRNA.csv", "datMeta_DNAm.csv")

# Inizializza la lista per salvare gli ID di ogni file datMeta
all_idx <- list()

# Itera su ciascun file e salva gli ID in all_idx
for (file in files) {
    # Costruisci il percorso completo del file
    file_path <- paste0(base_path, file)
    
    # Carica il file
    if (file.exists(file_path)) {
        datMeta <- read.csv(file_path, row.names = 1)
        all_idx[[file]] <- rownames(datMeta)
    } else {
        stop(paste("File non trovato:", file_path))
    }
}

# Trova gli ID comuni a tutti i file datMeta
common_ids <- Reduce(intersect, all_idx)

# Stampa gli ID comuni
print(paste("ID comuni trovati:", length(common_ids)))
print(common_ids)









# g <- snf.to.graph(W , datMeta , trait , all_idx , sub_mod_list)

# print(length(V(g)))
# write.csv(as_long_data_frame(g) , file = paste0('./data/',dataset,'/',project,'/raw/',paste0(sub_mod_list , collapse = '_'),'_graph.csv'))


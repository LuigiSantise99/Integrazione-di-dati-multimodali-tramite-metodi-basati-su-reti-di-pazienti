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

# Inizializza la lista per salvare gli ID di ogni file datMeta
idx <- list()

# Specifica i file datMeta da caricare
files <- c('../MOGDx/data/',dataset,'/',project,'/raw/datMeta_mRNA.csv', 
           '../MOGDx/data/',dataset,'/',project,'/raw/datMeta_miRNA.csv', 
           '../MOGDx/data/',dataset,'/',project,'/raw/datMeta_DNAm.csv')

# Itera su ciascun file e salva gli ID in all_idx
for (file in files) {
    # Carica il file
    datMeta <- read.csv(file, row.names = 1)
    
    # Salva gli ID dei pazienti (rownames) nella lista
    all_idx[[file]] <- rownames(datMeta)
}

# Trova gli ID comuni a tutti i file datMeta
common_ids <- Reduce(intersect, all_idx)

# Stampa gli ID comuni
print(paste("ID comuni trovati:", length(common_ids)))
print(common_ids)








# g <- snf.to.graph(W , datMeta , trait , all_idx , sub_mod_list)

# print(length(V(g)))
# write.csv(as_long_data_frame(g) , file = paste0('./data/',dataset,'/',project,'/raw/',paste0(sub_mod_list , collapse = '_'),'_graph.csv'))


library(reticulate)
library(jsonlite)
np <- import("numpy")
source('../MOGDx/PreProcessing/R_Preprocessing/preprocess_functions.R')

setwd('~/tesi-progetto')

trait <- c('paper_BRCA_Subtype_PAM50')
dataset <- 'TCGA'
project <- 'BRCA'
index_col <- 'patient'
modalities <- c( 'DNAm' , 'miRNA' , 'mRNA' )

W <- read.csv(file.path("affinity_matrices", project, "fused_affinity_matrix_with_id.csv"), row.names = 1) #ho usato snfpy

all_idx <- rownames(W)

colnames <- c('patient' ,  'race' , 'gender'  , 'ethnicity', 'age_at_diagnosis', trait)
datMeta <- t(data.frame( row.names = colnames))
for (mod in modalities) {
    print(mod)
    datMeta <- rbind(datMeta , read.csv(paste0('../MOGDx/data/',dataset,'/',project,'/raw/datMeta_',mod,'.csv') , row.names = 1)[ , colnames])
}
datMeta <- datMeta[!(duplicated(datMeta)),]
rownames(datMeta) <- datMeta[[index_col]]
print(dim(datMeta))

g <- snf.to.graph.fromPy(W , datMeta , all_idx, trait)
print(length(V(g)))


# Save the dataframe to CSV
write.csv(as_long_data_frame(g), file = file.path("graphsage_input", project, "exp_1", paste0(project, "-G.csv")))

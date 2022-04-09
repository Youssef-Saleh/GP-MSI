library(BiocManager)
#BiocManager::install("impute",force = TRUE)
library(samr)




setwd("C:/Users/omars/Desktop/GP-MSI/Breast Cancer")
joindata <- read.csv("BreastCancerAverageSpectrumValues.csv")
msdata <- joindata[,1:62]

d=list(x=t(msdata),y=joindata$Status, geneid=as.character(1:62),genenames=names(msdata),logged2=TRUE) 
set.seed(3)
samr.obj=samr(d, resp.type="Two class unpaired", nperms=200)
# samr.obj=samr(d, resp.type="Multiclass", nperms=200)
delta.table <- samr.compute.delta.table(samr.obj)

delta=  1.0564811363                #(delta.table) this is the value in delta.table(FDR <0.001)

samr.plot(samr.obj,delta)
siggenes.table<-samr.compute.siggenes.table(samr.obj,delta, d, delta.table)
siggenes.table

library(jsonlite)
significant_proteins <- subset(siggenes.table$genes.lo , select=c("Gene ID"))
write_json(significant_proteins, "significant_breast_proteins.json")

significant_proteins_up <- subset(siggenes.table$genes.up , select=c("Gene ID"))
write_json(significant_proteins_up, "significant_breast_proteins_up.json")

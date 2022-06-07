# Reading Imports
{
library(BiocManager)
#BiocManager::install("impute",force = TRUE)
library(samr)
}


# Running SAMR Algorithm and creating delta table
{
setwd("C:/Users/omars/Desktop/GP-MSI/Gastric Cancer")
joindata <- read.csv("GastricCancerAverageSpectrumValues.csv")
msdata <- joindata[,1:82]

d=list(x=t(msdata),y=joindata$Status, geneid=as.character(1:82),genenames=names(msdata),logged2=TRUE) 
set.seed(3)

samr.obj=samr(d, resp.type="Two class unpaired", nperms=200)
# samr.obj=samr(d, resp.type="Multiclass", nperms=200)
delta.table <- samr.compute.delta.table(samr.obj)
}

delta=   1.315954462     #(delta.table) this is the value in delta.table(FDR <0.001)


# Plot the significant proteins
{
samr.plot(samr.obj,delta)
siggenes.table<-samr.compute.siggenes.table(samr.obj,delta, d, delta.table)
siggenes.table
}

# Create two files for the significant proteins to be read in the model code
{
library(jsonlite)

Significant_Proteins_LO <- tryCatch(
  {
    significant_proteins <- subset(siggenes.table$genes.lo , select=c("Gene ID"))
    write_json(significant_proteins, "significant_gastric_proteins.json")
  },
  error = function(e){
    print(e)
    print('Error avoided by using empty list instead')
    significant_proteins <- list(list())
    write_json(significant_proteins, "significant_gastric_proteins.json")
  }
)
Significant_Proteins_LO


Significant_Proteins_HI <- tryCatch(
  {
    significant_proteins_up <- subset(siggenes.table$genes.up , select=c("Gene ID"))
    write_json(significant_proteins_up, "significant_gastric_proteins_up.json")
  },
  error = function(e){
    print(e)
    print('Error avoided by using empty list instead')
    significant_proteins_up <- list(list())
    write_json(significant_proteins_up, "significant_gastric_proteins_up.json")
  }
)

Significant_Proteins_HI
}
    

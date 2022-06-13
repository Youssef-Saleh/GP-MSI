#---------------------------------------------------------------------#
## IMPORTS-------------------------------------------------------------

# %matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import h5py

import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
from skimage.color import lab2rgb



import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.cluster import KMeans

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

import copy
import time

from sklearn.mixture import GaussianMixture 

#---------------------------------------------------------------------#
## Create Color Maps for 3D coloring-------------------------------------------------------------


from matplotlib.colors import LinearSegmentedColormap
def CreateColorMap(NumberofColors , colorsArray ):
    cmap = LinearSegmentedColormap.from_list('cmap', colorsArray, N=NumberofColors)
    return cmap

def CreateColorMap_Continuous(NumberofColors , colorsArray ):
    cvals = np.arange(0,NumberofColors-1)
    colors = colorsArray
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = LinearSegmentedColormap.from_list("", tuples)
    
    return cmap


#---------------------------------------------------------------------#
## Read Data function-------------------------------------------------------------


def ReadGastricData():
    h5 = h5py.File('gastricData.mat', 'r')


    HE_image= h5.get('HE_image')[:]
    HE_image = HE_image.swapaxes(0,2)


    MSI_data_cube = h5.get('MSI_data_cube')[:]
    MSI_data_cube = MSI_data_cube.swapaxes(0,2)

    goodlist= h5.get('goodlist')[:]
    goodlist = goodlist.swapaxes(0,1)

    peak_list= h5.get('peak_list')[:]
    peak_list = peak_list.swapaxes(0,1)

    pixel_to_sample_ID = h5.get('pixel_to_sample_ID')[:]
    pixel_to_sample_ID = pixel_to_sample_ID.swapaxes(0,1)  
    z = h5.get('z')[:]
    h5.close()

    # Clinical_data = pd.read_excel('ClinicalData.xlsx')

    flattened_MSI_data_cube = MSI_data_cube.flatten().reshape(MSI_data_cube.shape[0] * MSI_data_cube.shape[1] , MSI_data_cube.shape[2])

    flattened_pixel_to_sample_ID = pixel_to_sample_ID.flatten() 
    indices_of_background = np.where(flattened_pixel_to_sample_ID == -1)
    sample_only_data = np.delete(flattened_MSI_data_cube, indices_of_background[0] , axis=0)
    sample_ID_pixels = np.delete(flattened_pixel_to_sample_ID , indices_of_background[0] , axis=0)

    return HE_image , MSI_data_cube , goodlist, peak_list, pixel_to_sample_ID, sample_only_data , sample_ID_pixels


#---------------------------------------------------------------------#
## Split data according to LOPO, one patient is left out-------------------------------------------------------------


def SplitData(ID ,sample_only_data, sample_ID_pixels):

    test_patient_indicies=np.where(sample_ID_pixels == ID)

    mask = np.ones(sample_only_data.shape[0], dtype=bool)
    mask[test_patient_indicies] = False

    train_data = sample_only_data[mask]
    test_data = sample_only_data[test_patient_indicies]

    # The scaler object (model)
    scaler = StandardScaler()
    # fit and transform the data
    train_scaled_data = scaler.fit_transform(train_data) 

    # The scaler object (model)
    scaler = StandardScaler()
    # fit and transform the data
    test_scaled_data = scaler.fit_transform(test_data) 

    train_ID_pixels = sample_ID_pixels[sample_ID_pixels != ID]
    test_ID_pixels = sample_ID_pixels[sample_ID_pixels == ID]

    return train_scaled_data,train_data,train_ID_pixels,test_scaled_data,test_data,test_ID_pixels


#---------------------------------------------------------------------#
## Applying tSNE dimension reduction on data to 3D space-------------------------------------------------------------


def tSNE(sample_only_scaled_data):
    time_start = time.time()
    tsne_results_op = TSNE(n_components=3,
            perplexity = 50,
            learning_rate = 200,
            init = 'random',
            random_state = 0,
            early_exaggeration = 12,
            n_iter = 1000,
            verbose=True,
            ).fit_transform(sample_only_scaled_data)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    return tsne_results_op


#---------------------------------------------------------------------#
## Applying all required K-means clustering-------------------------------------------------------------


def KMeans_results(tsne_results):

    kmeans_3 = KMeans(n_clusters=3, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_4 = KMeans(n_clusters=4, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_5 = KMeans(n_clusters=5, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_6 = KMeans(n_clusters=6, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_7 = KMeans(n_clusters=7, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_8 = KMeans(n_clusters=8, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)

    return kmeans_3 , kmeans_4, kmeans_5, kmeans_6, kmeans_7, kmeans_8


#---------------------------------------------------------------------#
## Reading the ground truth of data-------------------------------------------------------------


def ReadClinicalData(ID, file):
    Clinical_data = pd.read_excel(file)
    Clinical_data = Clinical_data.drop(labels=ID-1,axis=0)
    Clinical_data = Clinical_data.reset_index(drop=True)

    return Clinical_data


#---------------------------------------------------------------------#
## Applying Survival function of kaplan meier curve, outputs the kaplan meier survival curve-------------------------------------------------------------


def KaplanMeierSurvivalFitter( labels , colors ,Clinical_data , sample_ID_pixels , SignCluster=[]):

    Clinical_data_copied = Clinical_data.copy(deep=True)
    

    labels_count = len(np.unique(labels))
    Clusters = [ [] for _ in range(labels_count) ]

    for i in range(1,len(Clinical_data)+1):
        Pixels_Samples = np.where(sample_ID_pixels == i)[0]
        Patient_Labels = labels[Pixels_Samples]

        for cluster_label in range(labels_count):
            Patient_Pixels = Patient_Labels[Patient_Labels == cluster_label]
            if len(Patient_Pixels) >= int( (1/labels_count * len(Patient_Labels))):
                Clusters[cluster_label].append(i)
    
    Belong_Clusters = [ [] for _ in range(labels_count) ]

    for i in range(1, len(Clinical_data)+1):
        for j in range(labels_count):
            if (i in Clusters[j]):
                Belong_Clusters[j].append(1)
            else:
                Belong_Clusters[j].append(0)

    for cluster_label in range(labels_count):
        Clinical_data_copied["Belong_Cluster_" + str(cluster_label+1)] = Belong_Clusters[cluster_label]
    

    kaplan_fitters = [ [] for _ in range(labels_count) ]
    axes = [ [] for _ in range(labels_count)]
    Clusters = [ [] for _ in range(labels_count) ]
    fig = plt.figure(figsize=(10, 5))

    if not SignCluster:

        for cluster_label in range(labels_count):
            kaplan_fitters[cluster_label] = KaplanMeierFitter() ## instantiate the class to create an object
            Clusters[cluster_label] = Clinical_data_copied.query("Belong_Cluster_" + str(cluster_label+1) + " == 1")
            kaplan_fitters[cluster_label].fit(Clusters[cluster_label]["Surv_time"], Clusters[cluster_label]["Surv_status"], label='Cluster ' + str(cluster_label+1))
            axes[cluster_label] = kaplan_fitters[cluster_label].plot(ci_show=False)


        for cluster_label in range(labels_count):
            legend = axes[cluster_label].get_legend()
            hl_dict = {handle.get_label(): handle for handle in legend.legendHandles}
            hl_dict['Cluster ' + str(cluster_label+1)].set_color(colors[cluster_label])
            axes[cluster_label].get_lines()[cluster_label].set_color(colors[cluster_label])
    else:

        for cluster_label in SignCluster:
            kaplan_fitters[cluster_label] = KaplanMeierFitter() ## instantiate the class to create an object
            Clusters[cluster_label] = Clinical_data_copied.query("Belong_Cluster_" + str(cluster_label+1) + " == 1")
            kaplan_fitters[cluster_label].fit(Clusters[cluster_label]["Surv_time"], Clusters[cluster_label]["Surv_status"], label='Cluster ' + str(cluster_label+1))
            axes[cluster_label] = kaplan_fitters[cluster_label].plot(ci_show=False)

        incr=0

        for cluster_label in SignCluster:
            legend = axes[cluster_label].get_legend()
            hl_dict = {handle.get_label(): handle for handle in legend.legendHandles}
            hl_dict['Cluster ' + str(cluster_label+1)].set_color(colors[cluster_label])
            axes[cluster_label].get_lines()[incr].set_color(colors[cluster_label])
            incr+=1
            
    plt.title("Kaplan Meier Graph")
    plt.xlabel('Survival time (month)')
    plt.ylabel('Probability of Survival')
    plt.ylim([0,1])
    plt.xlim([0,60])
    plt.tight_layout()

   




    return Clinical_data_copied

#---------------------------------------------------------------------#
## Applying Log Rank function to get p-value to compare between 2 clusters-------------------------------------------------------------


def LogRankTest_PrintValues(labels,Clinical_data, printResults = False):

   labels_count = len(np.unique(labels))
   Results = [ [ [] for _ in range(labels_count) ] for _ in range(labels_count) ]
   Clusters = [ [] for _ in range(labels_count) ]

   for cluster_label in range(labels_count):
      Clusters[cluster_label] = Clinical_data.query("Belong_Cluster_" + str(cluster_label+1) + " == 1")

   for cluster_label_main in range(labels_count):

      for cluster_label_secondary in range(labels_count):

         if cluster_label_main == cluster_label_secondary:
               Results[cluster_label_main][cluster_label_secondary] = None
         else:
               Results[cluster_label_main][cluster_label_secondary] = logrank_test(
               Clusters[cluster_label_main]["Surv_time"],
               Clusters[cluster_label_secondary]["Surv_time"],
               Clusters[cluster_label_main]["Surv_status"] , 
               Clusters[cluster_label_secondary]["Surv_status"])

               if printResults == True:
                  print("Cluster " + str(cluster_label_main+1) + " with Cluster " + str(cluster_label_secondary+1) )
                  Results[cluster_label_main][cluster_label_secondary].print_summary()
                  print("\n")
   return Results
                


#---------------------------------------------------------------------#
## Cox hazard function to show which cluster is most hazardous/most detrimental to survival-------------------------------------------------------------


def CoxHazardFitter(labels , Clinical_data):

    Clinical_data_coxHazard = Clinical_data.copy(deep=True)
    Clinical_data_coxHazard.drop(["Sample_ID","T","N","M"],inplace = True,axis=1)

    # labels_count = len(np.unique(labels))
    # Clusters = [ [] for _ in range(labels_count) ]

    # for cluster_label in range(labels_count):
    #     Clusters[cluster_label] = Clinical_data_coxHazard.query("Belong_Cluster_" + str(cluster_label+1) + " == 1")

    # for cluster_label in range(labels_count):
    #     Clinical_data_coxHazard.drop(["Belong_Cluster_" + str(cluster_label+1)],inplace = True,axis=1)
    #     Clinical_data_coxHazard["Dead in Cluster " + str(cluster_label+1)] = Clusters[cluster_label]["Surv_status"]


    # Clinical_data_coxHazard = Clinical_data_coxHazard.fillna(0)

    # Applying CoxHazard
    cph=CoxPHFitter(penalizer=0.001)

    cph.fit(Clinical_data_coxHazard, "Surv_time", "Surv_status")

    cph.plot(hazard_ratios=True)
    cph.print_summary()

    return cph



#---------------------------------------------------------------------#
## Cox hazard's bar plot-------------------------------------------------------------


def CoxHazardBarPlot( cph , clusters, colors, max_tick_value):

    hazard_ratio = cph.summary["exp(coef)"]
    hazard_ratio = hazard_ratio.to_numpy()
    
    cmap = CreateColorMap_Continuous(len(colors)+1,colors)

    rescale = lambda hazard_ratio: ((hazard_ratio - np.min(hazard_ratio)) / (np.max(hazard_ratio) - np.min(hazard_ratio)))
    labels_count = len(clusters)
    x = np.arange(1,labels_count + 1)
    y = [ ]
    for cluster_label in range(labels_count):
        y.append(len(clusters[cluster_label]))

    bar = plt.bar(x, y,color = cmap(rescale(hazard_ratio)))
    plt.xticks(np.arange(1,labels_count+1))
    plt.title('Number of patients per subpopulation')
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,np.max(hazard_ratio) ) )

    cbar = plt.colorbar(sm, aspect=10,shrink=0.9, pad=0.03)
    
    max_tick = max_tick_value # variable to be edited for max tick
    cbar.ax.tick_params(size=0)
    cbar.set_ticks([0,max_tick]) # Comment the colorbar lines and look at the tick values to find the max value tick to be edited
    cbar.ax.set_yticklabels(['Low','High'],weight='bold',fontsize=20) 
    cbar.set_label("Hazard",labelpad= -2)
 
    # plt.yticks([])
    plt.show()


#---------------------------------------------------------------------#
## Preparing the file needed to be read for SAM Analysis in R file-------------------------------------------------------------


def SAM_Analysis(labels, Clinical_data, sample_ID_pixels, hazardous_cluster_label,sample_only_data,peak_list):

    labels_count=len(np.unique(labels))
    indices_of_patients = [[] for _ in range(len(Clinical_data))]
    Sig_Cluster=[[] for _ in range(len(Clinical_data))]
    Status=[[] for _ in range(len(Clinical_data))]

    index = 0
    for i in Clinical_data["Sample_ID"].tolist():
        Pixels_Samples = np.where(sample_ID_pixels == i)[0]
        Patient_Labels = labels[Pixels_Samples]
        
        for cluster_label in range(labels_count):

            Patient_Pixels = Patient_Labels[Patient_Labels == cluster_label]

            if len(Patient_Pixels) >= int((1/labels_count * len(Patient_Labels))):

                if cluster_label != hazardous_cluster_label:
                    
                    Status[index].append(1)
                else:
                    Status[index].append(2)  


                Sig_Cluster[index].append(cluster_label)

        for j in range(0,len(Sig_Cluster[index])):
            
            indices=np.where(labels == Sig_Cluster[index][j])[0]

            # for element in indices:
            #     if element in Pixels_Samples:
                    
            #         indices_of_patients[i-1].append(element)

            indices_of_patients[index].extend(list(set(Pixels_Samples).intersection(indices)))
        
        index += 1
    
    Final_Status=[]
    for patient in Status:
        if 2 in patient:
            Final_Status.append(2)
        else:
            Final_Status.append(1)
    
    Unique_IDs = Clinical_data["Sample_ID"].tolist()
    Metastasis_Patients = [ ]
    Average_protein_values = [ ]

    for i in range(0,len(Unique_IDs)):
        Patient_MSI_values = sample_only_data[indices_of_patients[i]]
        Metastasis_Patients.append(Patient_MSI_values)
        Average_protein_values.append(np.average(Metastasis_Patients[i], axis=0))
    
    protein_dataframe = pd.DataFrame(Average_protein_values,columns=peak_list[:,0].astype(int))
    protein_dataframe["Status"] = Final_Status

    return protein_dataframe
    


#---------------------------------------------------------------------#
## Significant Clusters function able to return the p-value between the significant clusters and the indices of the significant clusters-------------------------------------------------------------

def SignificantClusters(labels, Results):

    labels_count = len(np.unique(labels))
    pvalue_list =   [ [] for _ in range(labels_count) ]
    min_pvalues = [ ]

    for i in range(labels_count):
        for j in range(labels_count):
            if i == j:
                pass
            else:
                pvalue_list[i].append(round(Results[i][j].p_value,2))

    for i in range(labels_count):
        min_pvalues.append(np.min(pvalue_list[i]))

    SignCluster = [ ]
    for i in range(len(min_pvalues)):
        if min_pvalues[i] == np.min(min_pvalues):
            SignCluster.append(i)

    pvalue = np.min(min_pvalues)
    return pvalue , SignCluster


#---------------------------------------------------------------------#
## Reads the output file from R and formats it in order to be returned as a list-------------------------------------------------------------


def readSignificantProteins(file , delete=False):
    import json
    with open(file) as f:
        proteins = json.load(f)


    edited_proteins = [ ]
    for protein in proteins:
        for string in protein:
            string = string[1::]
            string = int(string)
            edited_proteins.append(string)

    # To delete the file, in order to be not confused with old versions of same file
    if delete == True:
        import os
        os.remove(file)

    return edited_proteins


#---------------------------------------------------------------------#
## Creating the classifier's training labels-------------------------------------------------------------
# If any patient has a poor survival cluster, the patient's pixels are labeled as 1
# Otherwise, if the patient has a high survival cluster, then they are labeled as 3
# Otherwise, they are labeled as moderate survival with 2

def TargetLabelsCreation(labels , Clinical_data, sample_ID_pixels, hazardous_cluster_label , survival_cluster_label):

    labels_count = len(np.unique(labels))
    Clusters = [ [] for _ in range(labels_count) ]
    Target_labels=copy.deepcopy(labels)

    # Change to the correct clusters identified in the survival analysis (use index not actual value)
    Poor_survival_cluster = hazardous_cluster_label
    High_survival_cluster = survival_cluster_label

    Poor = 1
    Moderate = 2
    High = 3

    for i in range(1,len(Clinical_data)+1):
        Pixels_Samples = np.where(sample_ID_pixels == i)[0]
        Patient_Labels = labels[Pixels_Samples]

        for cluster_label in range(labels_count):
            Patient_Pixels = Patient_Labels[Patient_Labels == cluster_label]
            if len(Patient_Pixels) >= int( (1/labels_count * len(Patient_Labels))):
                Clusters[cluster_label].append(i)

    for i in Clusters[Poor_survival_cluster]:
        Pixels_Samples = np.where(sample_ID_pixels == i)[0]
        Target_labels[Pixels_Samples] = Poor

    Target_labels[Target_labels != Poor] = Moderate

    for i in Clusters[High_survival_cluster]:
        Pixels_Samples = np.where(sample_ID_pixels == i)[0]
        Target_labels[Pixels_Samples] = High

    return Target_labels


#---------------------------------------------------------------------#
## SVM classifier that trains on the input data and returns the prediction of test data-------------------------------------------------------------


def SVM(used_kernel, regularization_value ,train_data,train_labels,test_data):
    from sklearn import svm

    #Create a svm Classifier
    clf = svm.SVC(kernel=used_kernel , C = regularization_value, random_state = 0)

    #Train the model using the training sets
    clf.fit(train_data, train_labels)

    #Predict the response for test dataset
    y_pred = clf.predict(test_data)

    return y_pred


#---------------------------------------------------------------------#
## KNN classifier that trains on the input data and returns the prediction of test data-------------------------------------------------------------


def KNN(neighbours, train_data, train_labels, test_data):
    
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(n_neighbors=neighbours)

    # Train the model using the training sets
    model.fit(train_data,train_labels)

    predicted_labels= model.predict(test_data)

    return predicted_labels


#---------------------------------------------------------------------#
## Calculates the total probability for each type, takes the output of classifier-------------------------------------------------------------


def ProbabilityCalc(y_pred,Poor=1,Moderate=2,High=3):
    
    Probability_arr=np.unique(y_pred,return_counts=True)
    
    Total_Propability = 0
    Poor_Surv = 0
    Moderate_Surv = 0
    High_Surv = 0

    for Probability in Probability_arr[1]:
        Total_Propability += Probability

    increment = 0
    for Surv_label in Probability_arr[0]:
        if Surv_label == Poor:
            Poor_Surv = (Probability_arr[1][increment] / Total_Propability) * 100
        elif Surv_label == Moderate:
            Moderate_Surv = (Probability_arr[1][increment] / Total_Propability) * 100
        elif Surv_label == High:
            High_Surv = (Probability_arr[1][increment] / Total_Propability) * 100
        increment += 1

    print("Poor survival probability : {} \nModerate survival probability : {} \nHigh survival probability : {}".format(Poor_Surv,Moderate_Surv,High_Surv))

    return Poor_Surv , Moderate_Surv , High_Surv


#---------------------------------------------------------------------#
## Saves the test results in specified format and returns a dataframe ready to be saved-------------------------------------------------------------


def OutputDataframe(total_results_dataframe,patient_ID , Clinical_data, Poor_Surv, Moderate_Surv, High_Surv, no_of_clusters, SAM_protein):
    
    results_dataframe = {}
    results_dataframe["Patient to be predicted/left out"] = patient_ID
    results_dataframe["Poor Survival Subpopulation"] = Poor_Surv
    results_dataframe["Moderate Survival Subpopulation"] = Moderate_Surv
    results_dataframe["Good Survival Subpopulation"] = High_Surv
    results_dataframe["Surv(months)"] = Clinical_data["Surv_time"][patient_ID-1]
    results_dataframe["Survival Status"] = Clinical_data["Surv_status"][patient_ID-1]
    prediction = "Poor"

    if Poor_Surv <= 10:
        prediction = "High"
    elif Poor_Surv > 10 and Poor_Surv <= 50:
        prediction = "Moderate"
    else:
        prediction = "Poor"
        
    results_dataframe["Predicted Survivability"] = prediction
    results_dataframe["Number of Clusters"] = no_of_clusters
    results_dataframe["SAM Features for each tSNE run on new subset"] = "Significant Features : m/z = " + str(SAM_protein)

    results_dataframe = pd.DataFrame([results_dataframe])
    total_results_dataframe = total_results_dataframe.append(results_dataframe)
    return total_results_dataframe


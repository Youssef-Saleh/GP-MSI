#---------------------------------------------------------------------#
## IMPORTS-------------------------------------------------------------

# %matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import h5py

import pandas as pd
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from skimage.color import lab2rgb , rgb2lab

import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.cluster import KMeans

from scipy.stats import hypergeom,fisher_exact

from sklearn.mixture import GaussianMixture 

import copy
import time



#---------------------------------------------------------------------#
## Read Data function-------------------------------------------------------------

def ReadBreastData():
    h5 = h5py.File('breastData.mat', 'r')


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

    return HE_image , MSI_data_cube , goodlist, peak_list, pixel_to_sample_ID, sample_only_data, sample_ID_pixels


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

    kmeans_11 = KMeans(n_clusters=11, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_10 = KMeans(n_clusters=10, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_9 = KMeans(n_clusters=9, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_8 = KMeans(n_clusters=8, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_7 = KMeans(n_clusters=7, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_6 = KMeans(n_clusters=6, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_5 = KMeans(n_clusters=5, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_4 = KMeans(n_clusters=4, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)
    kmeans_3 = KMeans(n_clusters=3, n_init=100, max_iter=2000, random_state=0, verbose=0 ).fit(tsne_results)


    return kmeans_11 , kmeans_10, kmeans_9, kmeans_8 , kmeans_7, kmeans_6, kmeans_5 , kmeans_4 , kmeans_3

#---------------------------------------------------------------------#
## Reading the ground truth of data-------------------------------------------------------------


def ReadClinicalData(ID, file):
    Clinical_data = pd.read_excel(file,"Breast Cancer Clinical Data")
    Clinical_data = Clinical_data.drop(labels=ID-1,axis=0)
    Clinical_data = Clinical_data.reset_index(drop=True)

    return Clinical_data


#---------------------------------------------------------------------#
## Applying Metastasis Analysis function to compute the number of metastatic and nonmetastatic patients in each cluster------------------------------------------------------------


def metastasis_status(labels, Clinical_data, sample_ID_pixels):

    Clinical_data_copied = Clinical_data.copy(deep=True)
    labels_count = len(np.unique(labels))
    Clusters = [[] for _ in range(labels_count)]

    for i in Clinical_data["Sample_ID"].tolist():
        Pixels_Samples = np.where(sample_ID_pixels == i)[0]
        Patient_Labels = labels[Pixels_Samples]

        for cluster_label in range(labels_count):
            Patient_Pixels = Patient_Labels[Patient_Labels == cluster_label]
            if len(Patient_Pixels) >= int((1/labels_count * len(Patient_Labels))):
                Clusters[cluster_label].append(i)

    NonMetastasis_Clusters = [[] for _ in range(labels_count)]
    Metastasis_Clusters = [[] for _ in range(labels_count)]


    patient_id_iterator = 1
    for i in range(0, len(Clinical_data)):
        for j in range(labels_count):
            if (patient_id_iterator in Clusters[j]):
                if Clinical_data["pN"][i] == 1:
                    NonMetastasis_Clusters[j].append(1)
                elif Clinical_data["pN"][i] == 2:
                    Metastasis_Clusters[j].append(1)
        patient_id_iterator += 1


    return NonMetastasis_Clusters, Metastasis_Clusters



#---------------------------------------------------------------------#
## Plotting the metastatic and nonmetastatic patients of each cluster as a stacked barplot------------------------------------------------------------



def plot_metastasis_status(state0, state1):
   ind = len(state0)
   status0 = np.zeros(ind)
   status1 = np.zeros(ind)
   cluster_numbers = []
   for i in range(1, ind+1):
      cluster_numbers.append(str(i))
      

   for i in range(len(state0)):
      status0[i] = len(state0[i])
   for i in range(len(state1)):
      status1[i] = len(state1[i])
   print("NonMetastasis: " + str(status0))
   print("Metastasis: " + str(status1))

   plt.bar(cluster_numbers, status0, color="#d4d0c8")
   plt.bar(cluster_numbers, status1, bottom=status0, color="black")
   plt.xticks(cluster_numbers)
   plt.xlabel("Cluster numbers")
   plt.ylabel("Number of patients")
   plt.title("Metastasis Analysis of {} Clusters".format(ind))
   colors = {'pN0': '#d4d0c8', 'pN1': 'black'}
   labels = list(colors.keys())
   handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label])
              for label in labels]
   plt.legend(handles, labels)

   plt.show()



#---------------------------------------------------------------------#
## Preparing the file needed to be read for SAM Analysis in R file-------------------------------------------------------------

def SAM_Analysis(labels, Clinical_data, sample_ID_pixels, fully_metastasis_cluster_label,sample_only_data,peak_list):
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

                if cluster_label != fully_metastasis_cluster_label:
                    
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
# If any patient has a metastatic cluster, the patient's pixels are labeled as 2
# Otherwise, they are labeled as non-metastatic with 1



# Used as labels for the SVM/KNN models, turns all cluster labels in kmeans labels into 1 or 2 (based on breast data)
def TargetLabelsCreation(labels , Clinical_data, sample_ID_pixels, fully_metastasis_cluster_label):
    # score values for metastasis and non_metastasis
    Non_Metastasis = 1
    Metastasis = 2

    labels_count=len(np.unique(labels))
    Status=[[] for _ in range(len(Clinical_data))]

    index = 0
    for i in Clinical_data["Sample_ID"]:

        Pixels_Samples = np.where(sample_ID_pixels == i)[0]
        Patient_Labels = labels[Pixels_Samples]
        
        for cluster_label in range(labels_count):

            Patient_Pixels = Patient_Labels[Patient_Labels == cluster_label]

            if len(Patient_Pixels) >= int((1/labels_count * len(Patient_Labels))):

                if cluster_label != fully_metastasis_cluster_label:
                    Status[index].append(1)
                else:
                    Status[index].append(2)  
        index+=1

    Final_Status=[]
    for patient in Status:
        if 2 in patient:
            Final_Status.append(2)
        else:
            Final_Status.append(1)

    Target_labels=copy.deepcopy(labels)

    index = 0
    for i in Clinical_data["Sample_ID"]:
        if Final_Status[index] == 2:
            Pixels_Samples = np.where(sample_ID_pixels == i)[0]
            Target_labels[Pixels_Samples] = Metastasis
        index+=1
        
    Target_labels[Target_labels != Metastasis] = Non_Metastasis

    return Target_labels


#---------------------------------------------------------------------#
## SVM classifier that trains on the input data and returns the prediction of test data-------------------------------------------------------------


def SVM(used_kernel, regularization_value,train_data,train_labels,test_data):
    from sklearn import svm

    #Create a svm Classifier
    clf = svm.SVC(kernel=used_kernel , C = regularization_value, random_state=0)

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


def ProbabilityCalc(y_pred,Non_Metastasis=1,Metastasis=2):
    Probability_arr=np.unique(y_pred,return_counts=True)
    Total_Propability = 0
    Metastasis_prob = 0
    Non_Metastasis_prob = 0

    for Probability in Probability_arr[1]:
        Total_Propability += Probability

    increment = 0
    for Metastasis_label in Probability_arr[0]:
        if Metastasis_label == Non_Metastasis:
            Non_Metastasis_prob = (Probability_arr[1][increment] / Total_Propability) * 100
        elif Metastasis_label == Metastasis:
            Metastasis_prob = (Probability_arr[1][increment] / Total_Propability) * 100
        increment += 1

    print("Metastasis probability : {} \nNon-Metastasis probability : {}".format(Metastasis_prob,Non_Metastasis_prob))
    return Metastasis_prob , Non_Metastasis_prob


#---------------------------------------------------------------------#
## Saves the test results in specified format and returns a dataframe ready to be saved-------------------------------------------------------------


def OutputDataframe(total_results_dataframe,patient_ID , Clinical_data, Non_Metastasis_prob, Metastasis_prob, no_of_clusters,C_value, SAM_protein):
    results_dataframe = {}
    results_dataframe["Patient to be predicted/left out"] = patient_ID
    results_dataframe["Metastasis-Free Subpopulation"] = Non_Metastasis_prob
    results_dataframe["Metastasis Subpopulation"] = Metastasis_prob
    results_dataframe["Metastasis Status"] = Clinical_data["pN"][patient_ID-1]
    prediction = 0
    if Metastasis_prob <= 2:
        prediction = 1
    else:
        prediction = 2
    results_dataframe["Predicted Metastasis"] = prediction
    results_dataframe["Number of Clusters"] = no_of_clusters
    results_dataframe["SAM Features for each tSNE run on new subset"] = "Significant Features : m/z = " + str(SAM_protein)

    results_dataframe = pd.DataFrame([results_dataframe])
    total_results_dataframe = total_results_dataframe.append(results_dataframe)
    return total_results_dataframe


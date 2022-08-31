# The identification of cancer lesions in mammography images with missing pixels: analysis of morphology

This repository includes the original code of the article "The identification of cancer lesions in mammography images with missing pixels: analysis of morphology" presented in the Application Track of IEEE International Conference on Data Science and Advanced Analytics (DSAA 2022).


The research study of the article is divided into 3 phases (illustrated by the following image) and in this repository, the code is divided into 4 folders:
 1. Preprocessing
 2. Imputation Phase
 3. Morphology Phase
 4. Classification Phase.

<img width="498" alt="image" src="https://user-images.githubusercontent.com/57224933/187632144-87688d4e-6cff-4ad6-b55e-9d31690fd8d0.png">

## Preprocessing
In this folder, the code for the preprocessing of the images from the 4 different datasets is available. 

To setup the preprocessing of the datasets, the data must be downloaded using the link within the code file and the directory variable must be changed to identify the directory of the downloaded data.



## Imputation Phase
In this folder, the code for the imputation phase is available. The code is divided into base code, DIP code and CycleGAN code. The base code presents the code for the KNN, MICE, MC and VAE algorithms, and the remaining algorithms are in the other folders. There are separate scripts for ROI patch and whole mammography images.


## Morphology Phase
In this folder, the main code for the evaluation on the morphology is presented. 

## Classification Phase
In this folder, the main code for the evaluation on the classification is presented. 

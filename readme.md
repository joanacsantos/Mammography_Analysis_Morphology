# The identification of cancer lesions in mammography images with missing pixels: analysis of morphology

This repository includes the original code of the article "The identification of cancer lesions in mammography images with missing pixels: analysis of morphology" presented in the Application Track of IEEE International Conference on Data Science and Advanced Analytics (DSAA 2022).


The research study of the article is divided into 3 phases (illustrated by the following image) and in this repository, the code is divided into 4 folders:
 1. Preprocessing
 2. Imputation Phase
 3. Morphology Phase
 4. Classification Phase.

<img width="809" alt="Captura de ecrã 2022-09-01, às 09 27 42" src="https://user-images.githubusercontent.com/57224933/187869050-2d786c63-7bb5-43a7-89aa-584336478c8f.png">

## Preprocessing
In this folder, the code for the preprocessing of the images from the 4 different datasets is available. 

To setup the preprocessing of the datasets, the data must be downloaded using the link within the code file and the directory variable must be changed to identify the directory of the downloaded data.

## Imputation Phase
In this folder, the code for the imputation phase is available. The code is divided into base code, DIP code and CycleGAN code. The base code presents the code for the KNN, MICE, MC and VAE algorithms, and the remaining algorithms are in the other folders. There are separate scripts for ROI patch and whole mammography images.

## Morphology Phase
In this folder, the main code for the evaluation on the morphology is presented. Only the analysis for the entire dataset is included. To perform the analysis by morphology categories, the datasets must be separated into subsets of the images that require the original csv files from the CBIS-DDSM database.

## Classification Phase
In this folder, the main code for the evaluation on the classification is presented. The code for the classifiers is separated into 2 scripts: one for the benign and malignant classification and another for the calcification and mass classification. For the evaluation, both classifiers are included in the process, however, the scripts for KNN/MICE and DIP are separated.

# Quantifying ECG Deviations with Latent Space Features for Improved Classification Reliability

This github repository contains the code, trained model and data assosiated with this thesis. 

## Overview of files

The files:   
`code/utility_functions/data_functions.py`   
`code/utility_functions/deviation_detection_functions.py`   
`deviation_detection_main.ipynb`   
`train_and_inspect_model_main.ipynb`   

Were made during this project.


The other files and the PTB-XL data set were downloaded from https://github.com/helme/ecg_ptbxl_benchmarking,
which is a framework made around the PTB-XL data set that was used to load the PTB-XL data and train the models

the file:   
`code/utils/utils.py`    
was edited to have custom tasks for only loading the training data classes MI and NORM,
and another task for loading the other classes HYP, CD and STTC for testing as unseen classes.

The file:   
`code/experiments/scp_experiment.py`  
was edited to allow for using cuda, where data during training is moved to the GPU.


The code 15% data set in this thesis and folder can be found here https://zenodo.org/records/4916206

        
Pre extracted features and the trained models are saved in the output folder, 
which means that the code can be ran without training the model.

## Dependencies

The file:    
condaENV    

contains the enviorment variables from the latest anaconda enviorment used and can be loaded directly into anaconda to recreate the enviorment.      
This should be all that is needed to run the code, but:   

depending on the system and intalled drivers, you might need visual studio and c++ tools which are common dependencies for some Python packages.     
The code should work without having the CUDA tool kit installed, but to use CUDA, this kit or similair CUDA packages needs to be installed. (https://developer.nvidia.com/cuda-toolkit)


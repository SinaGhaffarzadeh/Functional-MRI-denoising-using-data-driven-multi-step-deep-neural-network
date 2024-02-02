# Functional MRI denoising Using Data-Driven Multi-Step Deep Neural Network 

# MSDNN

Here is an implementation of the MSDNN model for de-noising task-based functional MRI data. For more details about the project please use [this link](https://www.ismrm.org/23/program-files/D-19.htm) to access the paper. These codes are implemented on Google Colab and for Neural Networks models we use Tensorflow to develop objective function and Keras to develop representational space. Any library you need for this project is provided inside the codes.

![MSDNN flow chart](https://github.com/SinaGhaffarzadeh/Functional-MRI-denoising-using-data-driven-multi-step-deep-neural-network/blob/master/Images/MSDNN.jpg?raw=true)


# Data

In this implementation, we focused on Working-Memory fMRI data that you can download them from [Human connectome project (HCP)]( https://www.humanconnectome.org/) that each data consist of Structural, Rest and Task functional MRI data.
Although you can download these data from there, we upload some data in our [Google Drive]( https://drive.google.com/drive/folders/1mm7s33QjCnEfCb7ipdZEMv3kiV18ZozI?usp=drive_link) to download and use in denoising process. These data are including Simulated and preprocessed fMRI data (_rest_fMRI_Gaussian.mat_), Raw Structural MRI (_T1w.nii_), Real task-fMRI data (_tfMRI_WM_LR.nii_), and Regions (_rest_fMRI_regions.mat_).


<p align="center">
  <img src="https://github.com/SinaGhaffarzadeh/Functional-MRI-denoising-using-data-driven-multi-step-deep-neural-network/blob/master/Images/unique_.png" />
</p>


# How to train the model

## Step One
Before starting the training process, first of all, extract all `.rar` files in their path. after that, please download existing data from `Google Drive` and put `T1w.nii` in `Structural folder` , `tfMRI_WM_LR.nii` in `tfMRI folder`, and `rest_fMRI_Gaussian.mat & rest_fMRI_regions.mat` in `Data floder` where the our data should be in there.

## Step Two
In this framework we have two training parts (Denoising Simulated and Real tfMRI data).

### .Simulate tFMRI
If you want to simulate tfMRI data you should download rest-fMRI data from HCP and add it up to `fMRI foldor`. After that, you'll able to run `Codes/Preprocessing/HCP_Simulation_tfMRI.m` to construct the simulated tfMRI.
Although you are able to do this processing yourself, there `/Data/` we did carry out this process too. because the rest fMRI data was very large and we couldn't upload it to our Google Drive. So, you can use them in the next step.

### .Real data
If you want to use real tfMRI data in denoising process you should just run `Codes/Preprocessing/HCP_Real_tfMRI.m` to extract segments of Gray matter and non-Gray matter and some other neccessery information.

### Training
After carrying out one of these processes we can apply our de-noising approach (`MSDNN`) to that. To start de-noising process you should use `Codes/Train_And_Test/TrainTest.py`.
This code is able to compute preprocessing, training, testing, and evaluating our input data that you got them from .mat code.


























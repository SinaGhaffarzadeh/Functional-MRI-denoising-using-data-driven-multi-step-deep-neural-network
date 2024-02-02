# Functional MRI denoising Using Data-Driven Multi-Step Deep Neural Network 

# MSDNN

Here is an implementation of the MSDNN model for de-noising task based functional MRI data. For more detail about the project please use [this link](https://www.ismrm.org/23/program-files/D-19.htm) to access the paper. These codes are implemented on Google Colab and for Neural Networks models we use Tensorflow to develop objective function and Keras to develop representational space. Any library you need for this project is provided inside the codes.

![MSDNN flow chart](https://github.com/SinaGhaffarzadeh/Functional-MRI-denoising-using-data-driven-multi-step-deep-neural-network/blob/master/Images/MSDNN.jpg?raw=true)


# Data
In this implementation we focused on Working-Memory fMRI data that you can download them form [Human connectome project (HCP)]( https://www.humanconnectome.org/) that each data consist of Structural, Rest and Task functional MRI data.
Although you can download these data from there, we upload some data in our [google Drive]( https://drive.google.com/drive/folders/1mm7s33QjCnEfCb7ipdZEMv3kiV18ZozI?usp=drive_link) to download and use in denoising process. These data are including Simulated and preprocessed fMRI data (_rest_fMRI_Gaussian.mat_), Raw Structural MRI (_T1w.nii_), Real task-fMRI data (_tfMRI_WM_LR.nii_), and Regions (_rest_fMRI_regions.mat_).

![HCP logo](https://github.com/SinaGhaffarzadeh/Functional-MRI-denoising-using-data-driven-multi-step-deep-neural-network/blob/master/Images/HCP.jpg?raw=true) ![HCP logo](https://github.com/SinaGhaffarzadeh/Functional-MRI-denoising-using-data-driven-multi-step-deep-neural-network/blob/master/Images/google.jpg?raw=true)


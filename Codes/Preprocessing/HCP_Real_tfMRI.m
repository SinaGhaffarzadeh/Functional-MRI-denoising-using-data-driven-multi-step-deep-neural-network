


% For using this code we need some data that extracted by SPM. With using
% this tool we can co-registred structual data to fMRI and also extract
% Gray matter, White matter, and CSF from structural data.
% Our dataset obtained from Human Connectom Project (HCP) and for
% simulating t-fMRI we need to have rest fMRI and structual data.
% you can downlod any data from this dataset (www.humanconnectome.org).
% In our study we used working memory data to denoise them.

clear
clc

datadir = '...\Codes\Data\Raw_Data\tfMRI';
stracdir = '...\Codes\Data\Raw_Data\Structural';

TR = 0.72;
FWHM = 0;

% Gray matter, White matter, and CSF Segments of structural MRI data that
% extracted by SPM and co-registered each of them on fMRI data.
rc1T1 = load_untouch_nii([stracdir,'\rc1T1w_restore_brain.nii']);rc1 = rc1T1.img;
rc2T1 = load_untouch_nii([stracdir,'\rc2T1w_restore_brain.nii']);rc2 = rc2T1.img;
rc3T1 = load_untouch_nii([stracdir,'\rc3T1w_restore_brain.nii']);rc3 = rc3T1.img;
% Create non-Gray matter data using White matter and CSF segments
rc23 = rc2T1.img+rc3T1.img;

% Creating a design matrix using information obtained during imaging. these
% pieces of information are in each HCP t-fMRI dataset.
eprime_file = [datadir,'\WM_run2_TAB.txt'];
sync_file = [datadir,'\Sync.txt'];

[X_tgtlure,ref_TR,X_deltaT,ref_deltaT]=HCP_tgtlureDesignmatrix(eprime_file,sync_file);
contrast_tgtlure = [1;0;-1];

[X_block,ref,ref_conv,deltaT] = HCP_blockDesignmatrix(eprime_file,sync_file);
contrast_block = [-1;1;0];

% Reading fMRI data and changing its shape into a list
% with two-dimension with its mask
fMRIdatanii = load_untouch_nii([datadir,'\tfMRI_WM_LR.nii']);
fMRIdata = fMRIdatanii.img;
fMRIdata = permute(fMRIdata,[4,1,2,3]);
% Normalize and Select 390 time-point from 405 time-point
fMRIdata = fMRIdata(16:end,:,:,:);
mask = squeeze(fMRIdata(1,:,:,:))>0;
fMRIdata = fMRIdata(:,mask>0);
fMRIdata = detrend(fMRIdata,1,0);

% Using Gaussian filter for smoothing in Gray and non-Gray matter regions
src23 = imgaussfilt3(rc23,1);
src23(mask<=0) = 0;

rc1 = imgaussfilt3(rc1,1);
rc1(mask<=0) = 0;

% Make activity map ( after applying activity in our regions)
[Activity_Mask,Beta_tgtlure_af,Const_tgtlure_af]=Univaranalysis(zscore(X_tgtlure(16:end,:)),zscore(fMRIdata),mask,contrast_tgtlure);
fMRIdata(:,isnan(Activity_Mask(mask>0))==1) = [];
mask(isnan(Activity_Mask)==1) = 0;

figure
imshow(Activity_Mask(:,:,45),[]) 

figure
imagesc(Activity_Mask(:,:,45),[0.2,0.5])
colorbar

% We use a threshold of 50% to purify gray and non-gray matter areas
% 1D form
threshold_of_nonGM_1D = src23(mask>0) > 0.5 * max(src23(:));
threshold_of_GM_1D = rc1(mask>0) > 0.5 * max(rc1(:));
% 3D form
threshold_of_nonGM_3D= src23 > 0.5 * max(src23(:));
src23_thresholded = src23(threshold_of_nonGM_3D == 1);

threshold_of_GM_3D= rc1 > 0.5 * max(rc1(:));
rc1_thresholded = rc1(threshold_of_GM_3D == 1);

Activity_Mask_1D_array = Activity_Mask(mask>0);

NonGM_in_Active_area = Activity_Mask_1D_array(threshold_of_nonGM_1D==1);
GM_in_Active_area = Activity_Mask_1D_array(threshold_of_GM_1D==1);

% Activated Gray matter and non-Gray matter areas
fMRIdata_GM = fMRIdata(:,threshold_of_GM_1D==1);
fMRIdata_nonGM = fMRIdata(:,threshold_of_nonGM_1D==1);

save([datadir,'\Matlab_Output\',sub,'\task_based_fMRI_Gaussian.mat'],'fMRIdata',"fMRIdata_GM","fMRIdata_nonGM",'Activity_Mask_1D_array',...
    'threshold_of_nonGM_1D','threshold_of_GM_1D','NonGM_in_Active_area',...
    'GM_in_Active_area','Activity_Mask','threshold_of_nonGM_3D','src23_thresholded', ...
    'threshold_of_GM_3D','rc1_thresholded','X_tgtlure','mask','-v7.3');


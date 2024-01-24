

% In this file we try to add activity to 6 various regions with low SNR 
% signal to make our simulation data 
% For using this code we need some data that extracted by SPM. With using
% this tool we can co-registred structual data to fMRI and also extract
% Gray matter, White matter, and CSF from structural data.
% Our dataset obtained from Human Connectom Project (HCP) and for
% simulating t-fMRI we need to have rest fMRI and structual data.
% you can downlod any data from this dataset (www.humanconnectome.org).
% In our study we used working memory data to denoise them.

clear
clc

% Directory
datadir = '...\Codes\Data\Raw_Data\rfMRI';
stracdir = '...\Codes\Data\Raw_Data\Structural';
infdir = '...\Codes\Data\Raw_Data\tfMRI';

% initializing params
TR = 0.72;
FWHM = 0;
f = 0.2;

fMRIdata_insula=[];fMRIdata_anterior_cingulate_cortex=[];fMRIdata_frontal_inferior_gyrus=[];
fMRIdata_middel_temporal_gyrus = [];fMRIdata_middle_frontal=[];fMRIdata_precentral=[];

insula_region = [];anterior_cingulate_cortex_region=[];frontal_inferior_gyrus_region=[];
middel_temporal_gyrus_region = []; middle_frontal_region=[]; precentral_region=[];


% Gray matter, White matter, and CSF Segments of structural MRI data that
% extracted by SPM and co-registered each of them on fMRI data.
rc1T1 = load_untouch_nii([stracdir,'\rc1T1w_restore_brain.nii']);rc1 = rc1T1.img;
rc2T1 = load_untouch_nii([stracdir,'\rc2T1w_restore_brain.nii']);rc2 = rc2T1.img;
rc3T1 = load_untouch_nii([stracdir,'\rc3T1w_restore_brain.nii']);rc3 = rc3T1.img;
% Create non-Gray matter data using White matter and CSF segments
rc23 = rc2T1.img+rc3T1.img;

% Regions that we used to construct our simulated data.
% these regions were extracted by the WFY PickAtlas Tool.
region_1 = load([datadir,'\Six_region_masks','\','insula.mat']);insula = region_1.insula;
region_2 = load([datadir,'\Six_region_masks','\','anterior_cingulate_cortex.mat']);anterior_cingulate_cortex = region_2.anterior_cingulate_cortex;
region_3 = load([datadir,'\Six_region_masks','\','frontal_inferior_gyrus.mat']);frontal_inferior_gyrus = region_3.frontal_inferior_gyrus;
region_4 = load([datadir,'\Six_region_masks','\','middel_temporal_gyrus.mat']);middel_temporal_gyrus = region_4.middel_temporal_gyrus;
region_5 = load([datadir,'\Six_region_masks','\','middle_frontal.mat']);middle_frontal = region_5.middle_frontal;
region_6 = load([datadir,'\Six_region_masks','\','precentral.mat']);precentral = region_6.precentral;


% Creating a design matrix using information obtained during imaging. these
% pieces of information are in each HCP t-fMRI dataset which we used them 
% for our simulated data.
eprime_file = [infdir,'\WM_run2_TAB.txt'];
sync_file = [infdir,'\Sync.txt'];

[X_tgtlure,ref_TR,X_deltaT,ref_deltaT]=HCP_tgtlureDesignmatrix(eprime_file,sync_file);
contrast_tgtlure = [1;0;-1];

% Reading fMRI data and changing its shape into a list
% with two-dimension with its mask
fMRIdatanii = load_untouch_nii([datadir,'\rfMRI_REST1_RL.nii']);
fMRIdata = fMRIdatanii.img;
fMRIdata = permute(fMRIdata,[4,1,2,3]);
% Normalize and Select 390 time-point from 1200 time-point
fMRIdata = fMRIdata(201:590,:,:,:);
mask = squeeze(fMRIdata(1,:,:,:))>0;
fMRIdata = fMRIdata(:,mask>0);
% Remove polynomial trend and Normalize it
fMRIdata = detrend(fMRIdata,1,0);
fMRIdata = zscore(fMRIdata);
% Make primary activity map ( before applying activity in our regions)
[Cor_tgtlure]=Univaranalysis(zscore(X_tgtlure(16:end,:)),fMRIdata,mask,contrast_tgtlure);
fMRIdata(:,isnan(Cor_tgtlure(mask>0))==1) = [];
mask(isnan(Cor_tgtlure)==1) = 0;


All_6reg_mask = {insula(mask>0),anterior_cingulate_cortex(mask>0),frontal_inferior_gyrus(mask>0),...
    middel_temporal_gyrus(mask>0),middle_frontal(mask>0),precentral(mask>0)};

len_six_reg_fMRI = {size(fMRIdata(:,insula(mask>0)>0)),size(fMRIdata(:,anterior_cingulate_cortex(mask>0)>0)),size(fMRIdata(:,frontal_inferior_gyrus(mask>0)>0)),...
    size(fMRIdata(:,middel_temporal_gyrus(mask>0)>0)),size(fMRIdata(:,middle_frontal(mask>0)>0)),size(fMRIdata(:,precentral(mask>0)>0))};

All_6reg_fMRI = {fMRIdata_insula,fMRIdata_anterior_cingulate_cortex,fMRIdata_frontal_inferior_gyrus,...
    fMRIdata_middel_temporal_gyrus,fMRIdata_middle_frontal,fMRIdata_precentral};

All_6reg = {insula_region,anterior_cingulate_cortex_region,frontal_inferior_gyrus_region,...
    middel_temporal_gyrus_region,middle_frontal_region,precentral_region};


% % Make and Add simulated tasks for our regions
B(:,1) = [1,0,0];% An active area
B(:,2) = [0,1,0];% An active area
B(:,3) = [0,0,1];% An active area
B(:,4) = [0.3,1,0];% Two active area
B(:,5) = [0.45,0,0.95];% Two active area
B(:,6) = [1,0.3,0.3];% Three active area

for reg=1:6
    r = normrnd(0,1,[3,1]);
    B_had = B(:,reg) + 0.1 * r;
    y_region = X_tgtlure(16:end,:) * B_had;
    len_fM = len_six_reg_fMRI{reg};
        mod_reg = All_6reg{reg};
    for samelen=1:len_fM(2)
        mod_reg(:,samelen) = y_region;
    end
    All_6reg{reg} = mod_reg;
    All_6reg_fMRI{reg} = fMRIdata(:,All_6reg_mask{reg}>0) + f * mod_reg;
    fMRIdata(:,All_6reg_mask{reg}>0) = All_6reg_fMRI{reg};
end


% Using Gaussian filter for smoothing in Gray and non-Gray matter regions
src23 = imgaussfilt3(rc23,1);
src23(mask<=0) = 0;

rc1 = imgaussfilt3(rc1,1);
rc1(mask<=0) = 0;

% Make secondary activity map ( after applying activity in our regions)
[Activity_Mask,Beta_tgtlure_af,Const_tgtlure_af]=Univaranalysis(zscore(X_tgtlure(16:end,:)),fMRIdata,mask,contrast_tgtlure);
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

save([datadir,'\Matlab_Output\',sub,'\rest_fMRI_Gaussian.mat'],'fMRIdata',"fMRIdata_GM","fMRIdata_nonGM",'Activity_Mask_1D_array',...
    'threshold_of_nonGM_1D','threshold_of_GM_1D','NonGM_in_Active_area',...
    'GM_in_Active_area','Activity_Mask','threshold_of_nonGM_3D','src23_thresholded', ...
    'threshold_of_GM_3D','rc1_thresholded','X_tgtlure','mask','-v7.3');

save([datadir,'\Matlab_Output\',sub,'\rest_fMRI_regions.mat'],'mask','insula','anterior_cingulate_cortex', ...
    'frontal_inferior_gyrus','middel_temporal_gyrus','middle_frontal','precentral','-v7.3');


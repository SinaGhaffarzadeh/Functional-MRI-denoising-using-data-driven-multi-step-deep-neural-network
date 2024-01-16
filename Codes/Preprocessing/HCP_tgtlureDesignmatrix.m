function [X_TR,ref_TR,X_deltaT,ref_deltaT]=HCP_tgtlureDesignmatrix(eprime_file,sync_file)

TR=0.72;tdim = 405;taskduration = 2.5;
eprimedata = importdata(eprime_file);
eprimedata = eprimedata.textdata;
synctime = importdata(sync_file);

%%%find correct tab column
ind_stimONSET = find(strcmpi(eprimedata(1,:),'Stim.OnsetTime'));
ind_tgtlure = find(strcmpi(eprimedata(1,:),'TargetType'));

rowind_target = find(strcmpi(eprimedata(:,ind_tgtlure),'target'));
rowind_nonlure = find(strcmpi(eprimedata(:,ind_tgtlure),'nonlure'));
rowind_lure = find(strcmpi(eprimedata(:,ind_tgtlure),'lure'));

design_rowind = {rowind_target,rowind_nonlure,rowind_lure};
totaltime = TR*tdim;deltaT = 0.1;
ref_deltaT = zeros(round(totaltime/deltaT),numel(design_rowind));
X_deltaT = zeros(round(totaltime/deltaT),numel(design_rowind));

deltaT_array = deltaT*(1:round(totaltime/deltaT));
TR_array = TR*(1:tdim);canonHRF = spm_hrf(deltaT);

ref_TR = zeros(tdim,numel(design_rowind));
X_TR = zeros(tdim,numel(design_rowind));
for i = 1:numel(design_rowind)
    onsettime_temp = eprimedata(design_rowind{i},ind_stimONSET);
    for j = 1:numel(design_rowind{i})
        start_time = str2num(onsettime_temp{j})/1000;
        ref_deltaT(round((start_time-synctime)/deltaT):...
            round((start_time+taskduration-synctime)/deltaT),i) = 1;
    end
    temp = conv(ref_deltaT(:,i),canonHRF);
    X_deltaT(:,i) = temp(1:numel(deltaT_array));
    
    ref_TR(:,i) = interp1(deltaT_array',ref_deltaT(:,i),TR_array','PCHIP','extrap');
    X_TR(:,i) = interp1(deltaT_array',X_deltaT(:,i),TR_array,'pchip','extrap');
end


end

clear;
clc;
parent = 'IBRH13_data';
target = 'IBRH13_data';
dirlist = dir([parent '\*']);
parfor_progress(length(dirlist));
tic;
parfor i = 1 : length(dirlist)
    if dirlist(i).isdir && ~strcmp(dirlist(i).name, '.') && ~strcmp(dirlist(i).name, '..')
        try
            dh = load([parent '\' dirlist(i).name '\' dirlist(i).name 'EW_dh_005.acc']);
            up = load([parent '\' dirlist(i).name '\' dirlist(i).name 'EW_up_005.acc']);
            dt = 0.005;
        catch
            dh = load([parent '\' dirlist(i).name '\' dirlist(i).name 'EW_dh_010.acc']);
            up = load([parent '\' dirlist(i).name '\' dirlist(i).name 'EW_up_010.acc']);
            dt = 0.01;
        end
        dhf = acausal(20, 0.5, 2, dh, dt);
        upf = acausal(20, 0.5, 2, up, dt);
        acc_dh = fopen([target '\' dirlist(i).name '\' 'dhfilterEW.acc'], 'w');
        acc_up = fopen([target '\' dirlist(i).name '\' 'upfilterEW.acc'], 'w');
        for k = 1 : length(dhf)
            fprintf(acc_dh, '%7.6E\n', dhf(k));
            fprintf(acc_up, '%7.6E\n', upf(k));
        end
        fclose('all');                
    end  
    parfor_progress;
end
toc;
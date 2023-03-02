clear all
clc
format compact


for ds_rate=[50, 100, 200]
    fall_train_path=strcat('/home/jsw/FallDeFi/dataset/ds_train_data/ds_fall_train_', num2str(ds_rate),'/*.mat')
    nonfall_train_path=strcat('/home/jsw/FallDeFi/dataset/ds_train_data/ds_nonfall_train_', num2str(ds_rate),'/*.mat')
    
    fall_train_list = dir(fall_train_path);
    nonfall_train_list = dir(nonfall_train_path);
    
    data_collection_time = 10;
    sampling_rate = 1000;
    
    for w=1:length(fall_train_list)
%     for w=1:1
        fall_train_list(w).name
        ds_mat_file = load(fall_train_list(w).name);
        ds_data_ant1 = ds_mat_file.result(:,:,1); % 30 x 10000
        ds_data_ant2 = ds_mat_file.result(:,:,2);
        ds_data_ant3 = ds_mat_file.result(:,:,3);
    
        ds_data_wden_ant1=[];
        ds_data_wden_ant2=[];
        ds_data_wden_ant3=[];
        scal = 'sln';%'mln''one'  Use model assuming standard Gaussian white noise.
        for i = 1:30
            ds_data_wden_ant1(i,:) = wden(ds_data_ant1(i,:),'sqtwolog','s',scal,4,'sym3');
            ds_data_wden_ant2(i,:) = wden(ds_data_ant2(i,:),'sqtwolog','s',scal,4,'sym3');
            ds_data_wden_ant3(i,:) = wden(ds_data_ant3(i,:),'sqtwolog','s',scal,4,'sym3');
        end

        %% PCA    
        dsc_ant1 = ds_data_wden_ant1; %  dsc = 30 x 10000
        dsc_ant2 = ds_data_wden_ant2;
        dsc_ant3 = ds_data_wden_ant3;
        
        [PCs_ant1,~,eigenvalue_ant1] = PCA(dsc_ant1(1:30,:)',30); % PCs=10000x30, eigenvalue=30x1 30개로 분해된 PC와 그에 해당하는 eigenvalue
        [PCs_ant2,~,eigenvalue_ant2] = PCA(dsc_ant2(1:30,:)',30); 
        [PCs_ant3,~,eigenvalue_ant3] = PCA(dsc_ant3(1:30,:)',30); 

        % num of PCs = 10
        ds_wden_pca_ant1 = PCs_ant1(:, 1:10);
        ds_wden_pca_ant2 = PCs_ant2(:, 1:10);
        ds_wden_pca_ant3 = PCs_ant3(:, 1:10);

        result = cat(3, ds_wden_pca_ant1, ds_wden_pca_ant2, ds_wden_pca_ant3);
        size(result)
        
        new_name = extractBefore(fall_train_list(w).name, length(fall_train_list(w).name)-3) %remove '_interp.mat'
        file_name = strcat(new_name, '_wden_pca.mat')
        save(fullfile(pwd, 'dataset', 'ds_train_data', strcat('ds_fall_train_', num2str(ds_rate), '_wden_pca'), file_name), 'result');
        whos('-file',fullfile(pwd, 'dataset', 'ds_train_data', strcat('ds_fall_train_', num2str(ds_rate), '_wden_pca'), file_name)) 
    end

    for w=1:length(nonfall_train_list)
%     for w=1:1
        nonfall_train_list(w).name
        ds_mat_file = load(nonfall_train_list(w).name);
        ds_data_ant1 = ds_mat_file.result(:,:,1); % 30 x 10000
        ds_data_ant2 = ds_mat_file.result(:,:,2);
        ds_data_ant3 = ds_mat_file.result(:,:,3);
    
        ds_data_wden_ant1=[];
        ds_data_wden_ant2=[];
        ds_data_wden_ant3=[];
        scal = 'sln';%'mln''one'  Use model assuming standard Gaussian white noise.
        for i = 1:30
            ds_data_wden_ant1(i,:) = wden(ds_data_ant1(i,:),'sqtwolog','s',scal,4,'sym3');
            ds_data_wden_ant2(i,:) = wden(ds_data_ant2(i,:),'sqtwolog','s',scal,4,'sym3');
            ds_data_wden_ant3(i,:) = wden(ds_data_ant3(i,:),'sqtwolog','s',scal,4,'sym3');
        end

        %% PCA    
        dsc_ant1 = ds_data_wden_ant1; %  dsc = 30 x 10000
        dsc_ant2 = ds_data_wden_ant2;
        dsc_ant3 = ds_data_wden_ant3;
        
        [PCs_ant1,~,eigenvalue_ant1] = PCA(dsc_ant1(1:30,:)',30); % PCs=10000x30, eigenvalue=30x1 30개로 분해된 PC와 그에 해당하는 eigenvalue
        [PCs_ant2,~,eigenvalue_ant2] = PCA(dsc_ant2(1:30,:)',30); 
        [PCs_ant3,~,eigenvalue_ant3] = PCA(dsc_ant3(1:30,:)',30); 

        % num of PCs = 10
        ds_wden_pca_ant1 = PCs_ant1(:, 1:10);
        ds_wden_pca_ant2 = PCs_ant2(:, 1:10);
        ds_wden_pca_ant3 = PCs_ant3(:, 1:10);

        result = cat(3, ds_wden_pca_ant1, ds_wden_pca_ant2, ds_wden_pca_ant3);
        size(result)
        
        new_name = extractBefore(nonfall_train_list(w).name, length(nonfall_train_list(w).name)-3) %remove '_interp.mat'
        file_name = strcat(new_name, '_wden_pca.mat')
        save(fullfile(pwd, 'dataset', 'ds_train_data', strcat('ds_nonfall_train_', num2str(ds_rate), '_wden_pca'), file_name), 'result');
        whos('-file',fullfile(pwd, 'dataset', 'ds_train_data', strcat('ds_nonfall_train_', num2str(ds_rate), '_wden_pca'), file_name)) 
    end
end
clear all
clc
format compact

fall_train_interp_path='/home/jsw/FallDeFi/dataset/fall_train_interp/*.mat';
nonfall_train_interp_path='/home/jsw/FallDeFi/dataset/nonfall_train_interp/*.mat';

fall_train_interp_list = dir(fall_train_interp_path);
nonfall_train_interp_list = dir(nonfall_train_interp_path);


data_collection_time = 10;
sampling_rate = 1000;

% for w=1:length(fall_train_interp_list)
% % for w=1:1
%     fall_train_interp_list(w).name
%     interp_mat_file = load(fall_train_interp_list(w).name);
%     interp_data_ant1 = interp_mat_file.result(:,:,1); % 30 x 10000
%     interp_data_ant2 = interp_mat_file.result(:,:,2);
%     interp_data_ant3 = interp_mat_file.result(:,:,3);
%     for ds_rate=[50, 100, 200]
%         dsdata_ant1 = [];
%         dsdata_ant2 = [];
%         dsdata_ant3 = [];
%         for i=1:30
%             dsdata_ant1(i,:) = downsample(interp_data_ant1(i,:),sampling_rate/ds_rate);
%             dsdata_ant2(i,:) = downsample(interp_data_ant2(i,:),sampling_rate/ds_rate);
%             dsdata_ant3(i,:) = downsample(interp_data_ant3(i,:),sampling_rate/ds_rate);
%         end
%         result=cat(3, dsdata_ant1, dsdata_ant2, dsdata_ant3);
%         
%         ds_rate
%         size(interp_data_ant1)
%         size(dsdata_ant1)
% 
%         new_name = extractBefore(fall_train_interp_list(w).name, length(fall_train_interp_list(w).name)-10); %remove '_interp.mat'
%         file_name = strcat(new_name, '_' ,num2str(ds_rate),  '.mat');
%         save(fullfile(pwd, 'dataset', 'ds_train_data', strcat('ds_fall_train_', num2str(ds_rate)), file_name), 'result');
%         whos('-file',fullfile(pwd, 'dataset', 'ds_train_data/', strcat('ds_fall_train_', num2str(ds_rate)), file_name)) 
%     end
% end

for w=1:length(nonfall_train_interp_list)
% for w=1:1
    nonfall_train_interp_list(w).name
    interp_mat_file = load(nonfall_train_interp_list(w).name);
    interp_data_ant1 = interp_mat_file.result(:,:,1); % 30 x 10000
    interp_data_ant2 = interp_mat_file.result(:,:,2);
    interp_data_ant3 = interp_mat_file.result(:,:,3);
    for ds_rate=[50, 100, 200]
        dsdata_ant1 = [];
        dsdata_ant2 = [];
        dsdata_ant3 = [];
        for i=1:30
            dsdata_ant1(i,:) = downsample(interp_data_ant1(i,:),sampling_rate/ds_rate);
            dsdata_ant2(i,:) = downsample(interp_data_ant2(i,:),sampling_rate/ds_rate);
            dsdata_ant3(i,:) = downsample(interp_data_ant3(i,:),sampling_rate/ds_rate);
        end
        result=cat(3, dsdata_ant1, dsdata_ant2, dsdata_ant3);
        
        ds_rate
        size(interp_data_ant1)
        size(dsdata_ant1)

        new_name = extractBefore(nonfall_train_interp_list(w).name, length(nonfall_train_interp_list(w).name)-10); %remove '_interp.mat'
        file_name = strcat(new_name, '_' ,num2str(ds_rate), '.mat');
        save(fullfile(pwd, 'dataset', 'ds_train_data', strcat('ds_nonfall_train_', num2str(ds_rate)), file_name), 'result');
        whos('-file',fullfile(pwd, 'dataset', 'ds_train_data', strcat('ds_nonfall_train_', num2str(ds_rate)), file_name)) 
    end
end
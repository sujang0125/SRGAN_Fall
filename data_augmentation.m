clear all
clc
format compact

fall_train_interp_path='/home/jsw/FallDeFi/dataset/fall_train_interp/*.mat';
nonfall_train_interp_path='/home/jsw/FallDeFi/dataset/nonfall_train_interp/*.mat';

fall_train_interp_list = dir(fall_train_interp_path);
nonfall_train_interp_list = dir(nonfall_train_interp_path);


data_collection_time = 10;
sampling_rate = 1000;

for w=1:length(fall_train_interp_list)
% for w=1:1
    fall_train_interp_list(w).name
    interp_mat_file = load(fall_train_interp_list(w).name);
    interp_data_ant1 = interp_mat_file.result(:,:,1); % 30 x 10000
    interp_data_ant2 = interp_mat_file.result(:,:,2);
    interp_data_ant3 = interp_mat_file.result(:,:,3);

    for target_rate=[50, 100, 200]
        min_ds_idx = floor(sampling_rate/target_rate) + 20;
        max_ds_idx = floor(sampling_rate/target_rate);
        
        for ds_idx=min_ds_idx:-2:max_ds_idx    
            for phase=0:10:ds_idx-1
                dsdata_ant1 = [];
                dsdata_ant2 = [];
                dsdata_ant3 = [];
                aug_ant1 = [];
                aug_ant2 = [];
                aug_ant3 = [];
                for i=1:30
                    dsdata_ant1(i,:) = downsample(interp_data_ant1(i,:),ds_idx, phase);
                    dsdata_ant2(i,:) = downsample(interp_data_ant2(i,:),ds_idx, phase);
                    dsdata_ant3(i,:) = downsample(interp_data_ant3(i,:),ds_idx, phase);
                end

                for i=1:30
                    aug_ant1(i,:) = interp1(dsdata_ant1(i,:), 1:(size(dsdata_ant1(i,:),2)-1)/(target_rate*10-1):size(dsdata_ant1(i,:),2));
                    aug_ant2(i,:) = interp1(dsdata_ant2(i,:), 1:(size(dsdata_ant2(i,:),2)-1)/(target_rate*10-1):size(dsdata_ant2(i,:),2));
                    aug_ant3(i,:) = interp1(dsdata_ant3(i,:), 1:(size(dsdata_ant3(i,:),2)-1)/(target_rate*10-1):size(dsdata_ant3(i,:),2));
                end
                result=cat(3, aug_ant1, aug_ant2, aug_ant3);
                size(interp_data_ant1)
                size(dsdata_ant1)
                size(aug_ant1)
                size(result)
                    
                new_name = extractBefore(fall_train_interp_list(w).name, length(fall_train_interp_list(w).name)-10); %remove '_interp.mat'
                file_name = strcat(new_name, '_ph',num2str(phase, '%02d'), '_', num2str(ds_idx, '%02d'),'to', num2str(sampling_rate/target_rate, '%02d'), '.mat')
                save(fullfile(pwd, 'dataset', 'augmented', strcat('fall_tr_aug_', num2str(target_rate)), file_name), 'result');
%                 whos('-file',fullfile(pwd, 'dataset', 'augmented', strcat('fall_tr_aug_', num2str(target_rate)), file_name))
            end
        end
    end
end


for w=1:length(nonfall_train_interp_list)
% for w=1:1
    nonfall_train_interp_list(w).name
    interp_mat_file = load(nonfall_train_interp_list(w).name);
    interp_data_ant1 = interp_mat_file.result(:,:,1); % 30 x 10000
    interp_data_ant2 = interp_mat_file.result(:,:,2);
    interp_data_ant3 = interp_mat_file.result(:,:,3);

    for target_rate=[50, 100, 200]
        min_ds_idx = floor(sampling_rate/target_rate) + 20;
        max_ds_idx = floor(sampling_rate/target_rate);
        
        for ds_idx=min_ds_idx:-2:max_ds_idx    
            for phase=0:10:ds_idx-1
                dsdata_ant1 = [];
                dsdata_ant2 = [];
                dsdata_ant3 = [];
                aug_ant1 = [];
                aug_ant2 = [];
                aug_ant3 = [];
                for i=1:30
                    dsdata_ant1(i,:) = downsample(interp_data_ant1(i,:),ds_idx, phase);
                    dsdata_ant2(i,:) = downsample(interp_data_ant2(i,:),ds_idx, phase);
                    dsdata_ant3(i,:) = downsample(interp_data_ant3(i,:),ds_idx, phase);
                end

                for i=1:30
                    aug_ant1(i,:) = interp1(dsdata_ant1(i,:), 1:(size(dsdata_ant1(i,:),2)-1)/(target_rate*10-1):size(dsdata_ant1(i,:),2));
                    aug_ant2(i,:) = interp1(dsdata_ant2(i,:), 1:(size(dsdata_ant2(i,:),2)-1)/(target_rate*10-1):size(dsdata_ant2(i,:),2));
                    aug_ant3(i,:) = interp1(dsdata_ant3(i,:), 1:(size(dsdata_ant3(i,:),2)-1)/(target_rate*10-1):size(dsdata_ant3(i,:),2));
                end
                result=cat(3, aug_ant1, aug_ant2, aug_ant3);
                size(interp_data_ant1)
                size(dsdata_ant1)
                size(aug_ant1)
                size(result)
                    
                new_name = extractBefore(nonfall_train_interp_list(w).name, length(nonfall_train_interp_list(w).name)-10); %remove '_interp.mat'
                file_name = strcat(new_name, '_ph',num2str(phase, '%02d'), '_', num2str(ds_idx, '%02d'),'to', num2str(sampling_rate/target_rate, '%02d'), '.mat')
                save(fullfile(pwd, 'dataset', 'augmented', strcat('nonfall_tr_aug_', num2str(target_rate)), file_name), 'result');
%                 whos('-file',fullfile(pwd, 'dataset', 'augmented', strcat('nonfall_tr_aug_', num2str(target_rate)), file_name))
            end
        end
    end
end
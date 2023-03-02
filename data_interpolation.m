clear all
clc
format compact

fall_train_data_path='/home/jsw/FallDeFi/dataset/fall_train/*.dat';
nonfall_train_data_path='/home/jsw/FallDeFi/dataset/nonfall_train/*.dat';
fall_test_data_path='/home/jsw/FallDeFi/dataset/fall_test/*.dat';
nonfall_test_data_path='/home/jsw/FallDeFi/dataset/nonfall_test/*.dat';

fall_train_list = dir(fall_train_data_path);
nonfall_train_list = dir(nonfall_train_data_path);
fall_test_list = dir(fall_test_data_path);
nonfall_test_list = dir(nonfall_test_data_path);


for w=1:length(fall_train_list)
% for w=11:11
    fall_train_csi_trace = read_bf_file(fall_train_list(w).name);
    csiamp_ant1 = [];
    csiamp_ant2 = [];
    csiamp_ant3 = [];
    for i=1:length(fall_train_csi_trace)
        csi_entry = fall_train_csi_trace{i};
        csi = get_scaled_csi(csi_entry);
        % size(csi) = (1, 3, 30)
        
        %% Since we only have Rx 1, there are 3 antenna pairs
        csiamp_ant1 = [csiamp_ant1, abs(squeeze(csi(1,1,:)))];
        csiamp_ant2 = [csiamp_ant2, abs(squeeze(csi(1,2,:)))];
        csiamp_ant3 = [csiamp_ant3, abs(squeeze(csi(1,3,:)))];
    end
    size(csiamp_ant1)
    size(csiamp_ant2)
    size(csiamp_ant3)

    %% Linear Interpolation
    csi_length = 10000;
    for i=1:30
        csiamp_interp_ant1(i,:) = interp1(csiamp_ant1(i,:), 1:(size(csiamp_ant1(i,:),2)-1)/(csi_length - 1):size(csiamp_ant1(i,:),2));
        csiamp_interp_ant2(i,:) = interp1(csiamp_ant2(i,:), 1:(size(csiamp_ant2(i,:),2)-1)/(csi_length - 1):size(csiamp_ant2(i,:),2));
        csiamp_interp_ant3(i,:) = interp1(csiamp_ant3(i,:), 1:(size(csiamp_ant3(i,:),2)-1)/(csi_length - 1):size(csiamp_ant3(i,:),2));
    end

    result = cat(3,csiamp_interp_ant1,csiamp_interp_ant2,csiamp_interp_ant3);
    size(result)

    new_name = extractBefore(fall_train_list(w).name, length(fall_train_list(w).name)-3); %remove '.dat'
    file_name = strcat(new_name, '_interp');
    save(fullfile(pwd, '/dataset/fall_train_interp', file_name), 'result');
    whos('-file',fullfile(pwd, '/dataset/fall_train_interp', file_name))
    %fall_data = permute(fall_data, [3, 1, 2]);
end



for w=1:length(fall_test_list)
% for w=13:13
    fall_test_csi_trace = read_bf_file(fall_test_list(w).name);
    csiamp_ant1 = [];
    csiamp_ant2 = [];
    csiamp_ant3 = [];
    for i=1:length(fall_test_csi_trace)
        csi_entry = fall_test_csi_trace{i};
        csi = get_scaled_csi(csi_entry);
        % size(csi) = (1, 3, 30)
        
        %% Since we only have Rx 1, there are 3 antenna pairs
        csiamp_ant1 = [csiamp_ant1, abs(squeeze(csi(1,1,:)))];
        csiamp_ant2 = [csiamp_ant2, abs(squeeze(csi(1,2,:)))];
        csiamp_ant3 = [csiamp_ant3, abs(squeeze(csi(1,3,:)))];
    end
    size(csiamp_ant1)
    size(csiamp_ant2)
    size(csiamp_ant3)

    %% Linear Interpolation
    csi_length = 10000;
    for i=1:30
        csiamp_interp_ant1(i,:) = interp1(csiamp_ant1(i,:), 1:(size(csiamp_ant1(i,:),2)-1)/(csi_length - 1):size(csiamp_ant1(i,:),2));
        csiamp_interp_ant2(i,:) = interp1(csiamp_ant2(i,:), 1:(size(csiamp_ant2(i,:),2)-1)/(csi_length - 1):size(csiamp_ant2(i,:),2));
        csiamp_interp_ant3(i,:) = interp1(csiamp_ant3(i,:), 1:(size(csiamp_ant3(i,:),2)-1)/(csi_length - 1):size(csiamp_ant3(i,:),2));
    end

    result = cat(3,csiamp_interp_ant1,csiamp_interp_ant2,csiamp_interp_ant3);
    size(result)

    new_name = extractBefore(fall_test_list(w).name, length(fall_test_list(w).name)-3); %remove '.dat'
    file_name = strcat(new_name, '_interp');
    save(fullfile(pwd, '/dataset/fall_test_interp', file_name), 'result');
    whos('-file',fullfile(pwd, '/dataset/fall_test_interp', file_name))
    %fall_data = permute(fall_data, [3, 1, 2]);
end


for w=1:length(nonfall_train_list)
% for w=10:10
    nonfall_train_csi_trace = read_bf_file(nonfall_train_list(w).name);
    csiamp_ant1 = [];
    csiamp_ant2 = [];
    csiamp_ant3 = [];
    for i=1:length(nonfall_train_csi_trace)
        csi_entry = nonfall_train_csi_trace{i};
        csi = get_scaled_csi(csi_entry);
        % size(csi) = (1, 3, 30)
        
        %% Since we only have Rx 1, there are 3 antenna pairs
        csiamp_ant1 = [csiamp_ant1, abs(squeeze(csi(1,1,:)))];
        csiamp_ant2 = [csiamp_ant2, abs(squeeze(csi(1,2,:)))];
        csiamp_ant3 = [csiamp_ant3, abs(squeeze(csi(1,3,:)))];
    end
    size(csiamp_ant1)
    size(csiamp_ant2)
    size(csiamp_ant3)

    %% Linear Interpolation
    csi_length = 10000;
    for i=1:30
        csiamp_interp_ant1(i,:) = interp1(csiamp_ant1(i,:), 1:(size(csiamp_ant1(i,:),2)-1)/(csi_length - 1):size(csiamp_ant1(i,:),2));
        csiamp_interp_ant2(i,:) = interp1(csiamp_ant2(i,:), 1:(size(csiamp_ant2(i,:),2)-1)/(csi_length - 1):size(csiamp_ant2(i,:),2));
        csiamp_interp_ant3(i,:) = interp1(csiamp_ant3(i,:), 1:(size(csiamp_ant3(i,:),2)-1)/(csi_length - 1):size(csiamp_ant3(i,:),2));
    end

    result = cat(3,csiamp_interp_ant1,csiamp_interp_ant2,csiamp_interp_ant3);
    size(result)

    new_name = extractBefore(nonfall_train_list(w).name, length(nonfall_train_list(w).name)-3); %remove '.dat'
    file_name = strcat(new_name, '_interp');
    save(fullfile(pwd, '/dataset/nonfall_train_interp', file_name), 'result');
    whos('-file',fullfile(pwd, '/dataset/nonfall_train_interp', file_name))
    %fall_data = permute(fall_data, [3, 1, 2]);
end


for w=1:length(nonfall_test_list)
% for w=2:2
    nonfall_test_csi_trace = read_bf_file(nonfall_test_list(w).name);
    csiamp_ant1 = [];
    csiamp_ant2 = [];
    csiamp_ant3 = [];
    for i=1:length(nonfall_test_csi_trace)
        csi_entry = nonfall_test_csi_trace{i};
        csi = get_scaled_csi(csi_entry);
        % size(csi) = (1, 3, 30)
        
        %% Since we only have Rx 1, there are 3 antenna pairs
        csiamp_ant1 = [csiamp_ant1, abs(squeeze(csi(1,1,:)))];
        csiamp_ant2 = [csiamp_ant2, abs(squeeze(csi(1,2,:)))];
        csiamp_ant3 = [csiamp_ant3, abs(squeeze(csi(1,3,:)))];
    end
    size(csiamp_ant1)
    size(csiamp_ant2)
    size(csiamp_ant3)

    %% Linear Interpolation
    csi_length = 10000;
    for i=1:30
        csiamp_interp_ant1(i,:) = interp1(csiamp_ant1(i,:), 1:(size(csiamp_ant1(i,:),2)-1)/(csi_length - 1):size(csiamp_ant1(i,:),2));
        csiamp_interp_ant2(i,:) = interp1(csiamp_ant2(i,:), 1:(size(csiamp_ant2(i,:),2)-1)/(csi_length - 1):size(csiamp_ant2(i,:),2));
        csiamp_interp_ant3(i,:) = interp1(csiamp_ant3(i,:), 1:(size(csiamp_ant3(i,:),2)-1)/(csi_length - 1):size(csiamp_ant3(i,:),2));
    end

    result = cat(3,csiamp_interp_ant1,csiamp_interp_ant2,csiamp_interp_ant3);
    size(result)

    new_name = extractBefore(nonfall_test_list(w).name, length(nonfall_test_list(w).name)-3); %remove '.dat'
    file_name = strcat(new_name, '_interp');
    save(fullfile(pwd, '/dataset/nonfall_test_interp', file_name), 'result');
    whos('-file',fullfile(pwd, '/dataset/nonfall_test_interp', file_name))
    %fall_data = permute(fall_data, [3, 1, 2]);
end


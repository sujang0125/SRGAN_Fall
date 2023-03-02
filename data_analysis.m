clear all
clc
format compact

fall_data_path='c:/FallDeFi/fall/*.dat';
nonfall_data_path='c:/FallDeFi/nonfall/*.dat';

fall_list = dir(fall_data_path);
nonfall_list = dir(nonfall_data_path);
fall_data = [];
for w=1:30
    fall_csi_trace = read_bf_file(fall_list(w).name);
    csiamp_ant1 = [];
    for i=1:length(fall_csi_trace)
        csi_entry = fall_csi_trace{i};
        csi = get_scaled_csi(csi_entry);
        % size(csi) = (1, 3, 30)
        
        %% Since we only have Rx 1, there are 3 antenna pairs
        csiamp_ant1 = [csiamp_ant1, abs(squeeze(csi(1,1,:)))];
    end
    size(csiamp_ant1)

    %% Linear Interpolation
    csi_length = 10000
    for i=1:30
        csiamp_interp_ant1(i,:) = interp1(csiamp_ant1(i,:), 1:(size(csiamp_ant1(i,:),2)-1)/(csi_length - 1):size(csiamp_ant1(i,:),2));
    end
    
    %% wavelet denoising not butterworth lpf
    scal = 'sln';%'mln''one'  Use model assuming standard Gaussian white noise.
    for i = 1:30
        csiamp_interp_wden_ant1(i,:) = wden(csiamp_interp_ant1(i,:),'sqtwolog','s',scal,4,'sym3');
    end
    x1=1:1:length(csiamp_interp_wden_ant1(20,:));
    y1=csiamp_interp_wden_ant1;
    figure
    plot(x1,y1)

end

nonfall_data = [];
for w=1:30
    fall_csi_trace = read_bf_file(nonfall_list(w).name);
    csiamp_ant1 = [];
    for i=1:length(fall_csi_trace)
        csi_entry = fall_csi_trace{i};
        csi = get_scaled_csi(csi_entry);
        % size(csi) = (1, 3, 30)
        
        %% Since we only have Rx 1, there are 3 antenna pairs
        csiamp_ant1 = [csiamp_ant1, abs(squeeze(csi(1,1,:)))];
    end
    size(csiamp_ant1)

    %% Linear Interpolation
    csi_length = 10000
    for i=1:30
        csiamp_interp_ant1(i,:) = interp1(csiamp_ant1(i,:), 1:(size(csiamp_ant1(i,:),2)-1)/(csi_length - 1):size(csiamp_ant1(i,:),2));
    end
    
    %% wavelet denoising not butterworth lpf
    scal = 'sln';%'mln''one'  Use model assuming standard Gaussian white noise.
    for i = 1:30
        csiamp_interp_wden_ant1(i,:) = wden(csiamp_interp_ant1(i,:),'sqtwolog','s',scal,4,'sym3');
    end
    x2=1:1:length(csiamp_interp_wden_ant1(20,:));
    y2=csiamp_interp_wden_ant1;
    figure
    plot(x2,y2)

end






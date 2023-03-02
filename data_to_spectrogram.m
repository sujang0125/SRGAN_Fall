clear all
clc
format compact

fall_data_path='c:/FallDeFi/dataset/falldata_amp_interp/*.mat';
nonfall_data_path='c:/FallDeFi/dataset/nonfalldata_amp_interp/*.mat';

fall_list = dir(fall_data_path);
nonfall_list = dir(nonfall_data_path);


data_collection_time = 10;
sampling_rate = 1000;

fall_data = [];

% for w=1:length(fall_list)
for w=1:3
    fall_list(w).name
    interp_mat_file = load(fall_list(w).name);
    interp_data_ant1 = interp_mat_file.fall_data_amp_interp(:,:,1); % 30 x 10000
    interp_data_ant2 = interp_mat_file.fall_data_amp_interp(:,:,2);
    interp_data_ant3 = interp_mat_file.fall_data_amp_interp(:,:,3);

    %% Wavelet Denoising
    scal = 'sln';%'mln''one'  Use model assuming standard Gaussian white noise.
        
    for i = 1:30
        interp_wden_ant1 = wden(occu_amp_RX12_centered(i,:),'sqtwolog','s',scal,4,'sym3');
        % interp_wden_ant1(i,:) = wden(interp_data_ant1(i,:),'sqtwolog','s',scal,4,'sym3');
        % interp_wden_ant2(i,:) = wden(interp_data_ant2(i,:),'sqtwolog','s',scal,4,'sym3');
        % interp_wden_ant3(i,:) = wden(interp_data_ant3(i,:),'sqtwolog','s',scal,4,'sym3');
    end

    %% PCA    
    size(interp_wden_ant1)

    dsc_ant1 = interp_wden_ant1; %  dsc = 30 x 10000
%     dsc_ant2 = interp_wden_ant2;
%     dsc_ant3 = interp_wden_ant3;
    
    [PCs_ant1,~,eigenvalue_ant1] = PCA(dsc_ant1(1:30,:)',30); % PCs=10000x30, eigenvalue=30x1 30개로 분해된 PC와 그에 해당하는 eigenvalue
%     [PCs_ant2,~,eigenvalue_ant2] = PCA(dsc_ant2(1:30,:)',30); 
%     [PCs_ant3,~,eigenvalue_ant3] = PCA(dsc_ant3(1:30,:)',30); 


    %% STFT
    fs=1000; nfft = 256;noverlap=256;window=512;
    % frequency (bin) resolution = sample_rate/fft_size= 1.953Hz, time resolution =
    % (wind-noverlap)/fs=.112s , Hamming window
    % as overlap increases processing time increases and noise immunity
    % The length of the FFT is a tradeoff between frequency and time resolution
    % stft_netcell_array  = NET.createArray('System.Double[][]',fall_cell_array_size);
    
    
    denoised_sig_ant1=PCs_ant1;
%     denoised_sig_ant2=PCs_ant2;
%     denoised_sig_ant3=PCs_ant3;
    %     test=denoised_sig(1,:);
    %     test  = stft_cell_array{i}';
    sum_S1 = 0;
%     sum_S2 = 0;
%     sum_S3 = 0;
    for ii = 1:10%num_PCs(i)
        [S1,~,~] = spectrogram(denoised_sig_ant1(:,ii),window,noverlap,nfft,fs);%/std(test(ii,:),0,2)
%         [S2,~,~] = spectrogram(denoised_sig_ant2(:,ii),window,noverlap,nfft,fs);%/std(test(ii,:),0,2)
%         [S3,~,~] = spectrogram(denoised_sig_ant3(:,ii),window,noverlap,nfft,fs);%/std(test(ii,:),0,2)
        sum_S1 = abs(S1)+sum_S1; 
%         sum_S2 = abs(S2)+sum_S2; 
%         sum_S3 = abs(S3)+sum_S3; 
    end    
%     size(sum_S3)
%     figure
    image(sum_S1)
    colorbar
    figure
%     image(sum_S2)
%     colorbar
%     figure
%     image(sum_S3)
%     colorbar
    stft_array_ant1 = imresize(sum_S1/10,[100 10000]);
%     stft_array_ant2 = imresize(sum_S2/10,[100 10000]);
%     stft_array_ant3 = imresize(sum_S3/10,[100 10000]);

    
end





clear all
clc
format compact

fall_train_interp_path='/home/jsw/FallDeFi/dataset/fall_train_interp/*.mat';
nonfall_train_interp_path='/home/jsw/FallDeFi/dataset/nonfall_train_interp/*.mat';
fall_test_interp_path='/home/jsw/FallDeFi/dataset/fall_test_interp/*.mat';
nonfall_test_interp_path='/home/jsw/FallDeFi/dataset/nonfall_test_interp/*.mat';

fall_train_interp_list = dir(fall_train_interp_path);
nonfall_train_interp_list = dir(nonfall_train_interp_path);
fall_test_interp_list = dir(fall_test_interp_path);
nonfall_test_interp_list = dir(nonfall_test_interp_path);


% for w=1:length(fall_train_interp_list)
for w=13:13

    fall_train_interp_list(w).name
    interp_mat_file = load(fall_train_interp_list(w).name);
    interp_data_ant1 = interp_mat_file.result(:,:,1); % 30 x 10000
    interp_data_ant2 = interp_mat_file.result(:,:,2);
    interp_data_ant3 = interp_mat_file.result(:,:,3);


    x1=1:1:length(interp_data_ant1(10,4501:5000));
    y1=interp_data_ant1(:,4501:5000);
    f = figure;
    plot(x1,y1)
    xlabel('packet index', 'FontSize', 15)
    ylabel('amplitude', 'FontSize', 15)
    title('Raw CSI', 'FontSize', 18)
%     f.Position(3:4) = [560 420];
    
    %% wavelet denoising
    scal = 'sln';%'mln''one'  Use model assuming standard Gaussian white noise.

    for i = 1:30
        csiamp_interp_wden_ant1(i,:) = wden(interp_data_ant1(i,:),'sqtwolog','s',scal,4,'sym3');
        csiamp_interp_wden_ant2(i,:) = wden(interp_data_ant2(i,:),'sqtwolog','s',scal,4,'sym3');
        csiamp_interp_wden_ant3(i,:) = wden(interp_data_ant3(i,:),'sqtwolog','s',scal,4,'sym3');
    end
    x2=1:1:length(csiamp_interp_wden_ant1(10,4501:5000));
    y2=csiamp_interp_wden_ant1(:,4501:5000);
    f = figure;
    plot(x2,y2)
    xlabel('packet index', 'FontSize', 15)
    ylabel('amplitude', 'FontSize', 15)
    title('after DWT', 'FontSize', 18)
%     f.Position(3:4) = [560 420];

    %% PCA    

    dsc_ant1 = csiamp_interp_wden_ant1; %  dsc = 30 x 10000
    dsc_ant2 = csiamp_interp_wden_ant2;
    dsc_ant3 = csiamp_interp_wden_ant3;
    
    [PCs_ant1,~,eigenvalue_ant1] = PCA(dsc_ant1(1:30,:)',30); % PCs=10000x30, eigenvalue=30x1 30개로 분해된 PC와 그에 해당하는 eigenvalue
    [PCs_ant2,~,eigenvalue_ant2] = PCA(dsc_ant2(1:30,:)',30); 
    [PCs_ant3,~,eigenvalue_ant3] = PCA(dsc_ant3(1:30,:)',30); 

    % num of PCs = 10
    csiamp_interp_wden_pca_ant1 = PCs_ant1(:, 1:10);
    csiamp_interp_wden_pca_ant2 = PCs_ant2(:, 1:10);
    csiamp_interp_wden_pca_ant3 = PCs_ant3(:, 1:10);
    
    x3=1:1:length(csiamp_interp_wden_pca_ant1(4501:5000,10));
    y3=csiamp_interp_wden_pca_ant1(4501:5000,:);
    f = figure;
    plot(x3,y3)
    xlabel('packet index', 'FontSize', 15)
    ylabel('PC value', 'FontSize', 15)
    title('after PCA', 'FontSize', 18)
%     f.Position(3:4) = [560 420];

    result = cat(3, csiamp_interp_wden_pca_ant1, csiamp_interp_wden_pca_ant2, csiamp_interp_wden_pca_ant3);
    size(result)

    new_name = extractBefore(fall_train_interp_list(w).name, length(fall_train_interp_list(w).name)-10); %remove '.dat'
    file_name = strcat(new_name, '_wden_pca');
%     save(fullfile(pwd, '/dataset/fall_train_wden_pca', file_name), 'result');
%     whos('-file',fullfile(pwd, '/dataset/fall_train_wden_pca', file_name))
    %fall_data = permute(fall_data, [3, 1, 2]);
end
% 
% for w=1:length(fall_test_interp_list)
% % for w=1:1
% 
%     fall_test_interp_list(w).name
%     interp_mat_file = load(fall_test_interp_list(w).name);
%     interp_data_ant1 = interp_mat_file.result(:,:,1); % 30 x 10000
%     interp_data_ant2 = interp_mat_file.result(:,:,2);
%     interp_data_ant3 = interp_mat_file.result(:,:,3);
% 
% % 
% %     x1=1:1:length(interp_data_ant1(10,:));
% %     y1=interp_data_ant1;
% %     figure
% %     plot(x1,y1)
%     
%     %% wavelet denoising
%     scal = 'sln';%'mln''one'  Use model assuming standard Gaussian white noise.
% 
%     for i = 1:30
%         csiamp_interp_wden_ant1(i,:) = wden(interp_data_ant1(i,:),'sqtwolog','s',scal,4,'sym3');
%         csiamp_interp_wden_ant2(i,:) = wden(interp_data_ant2(i,:),'sqtwolog','s',scal,4,'sym3');
%         csiamp_interp_wden_ant3(i,:) = wden(interp_data_ant3(i,:),'sqtwolog','s',scal,4,'sym3');
%     end
% 
%     %% PCA    
% 
%     dsc_ant1 = csiamp_interp_wden_ant1; %  dsc = 30 x 10000
%     dsc_ant2 = csiamp_interp_wden_ant2;
%     dsc_ant3 = csiamp_interp_wden_ant3;
%     
%     [PCs_ant1,~,eigenvalue_ant1] = PCA(dsc_ant1(1:30,:)',30); % PCs=10000x30, eigenvalue=30x1 30개로 분해된 PC와 그에 해당하는 eigenvalue
%     [PCs_ant2,~,eigenvalue_ant2] = PCA(dsc_ant2(1:30,:)',30); 
%     [PCs_ant3,~,eigenvalue_ant3] = PCA(dsc_ant3(1:30,:)',30); 
% 
%     % num of PCs = 10
%     csiamp_interp_wden_pca_ant1 = PCs_ant1(:, 1:10);
%     csiamp_interp_wden_pca_ant2 = PCs_ant2(:, 1:10);
%     csiamp_interp_wden_pca_ant3 = PCs_ant3(:, 1:10);
% 
%     result = cat(3, csiamp_interp_wden_pca_ant1, csiamp_interp_wden_pca_ant2, csiamp_interp_wden_pca_ant3);
%     size(result)
% 
%     new_name = extractBefore(fall_test_interp_list(w).name, length(fall_test_interp_list(w).name)-10); %remove '.dat'
%     file_name = strcat(new_name, '_wden_pca');
%     save(fullfile(pwd, '/dataset/fall_test_wden_pca', file_name), 'result');
%     whos('-file',fullfile(pwd, '/dataset/fall_test_wden_pca', file_name))
%     %fall_data = permute(fall_data, [3, 1, 2]);
% end
% 
% 
% 
% for w=1:length(nonfall_train_interp_list)
% % for w=1:1
% 
%     nonfall_train_interp_list(w).name
%     interp_mat_file = load(nonfall_train_interp_list(w).name);
%     interp_data_ant1 = interp_mat_file.result(:,:,1); % 30 x 10000
%     interp_data_ant2 = interp_mat_file.result(:,:,2);
%     interp_data_ant3 = interp_mat_file.result(:,:,3);
% 
% % 
% %     x1=1:1:length(interp_data_ant1(10,:));
% %     y1=interp_data_ant1;
% %     figure
% %     plot(x1,y1)
%     
%     %% wavelet denoising
%     scal = 'sln';%'mln''one'  Use model assuming standard Gaussian white noise.
% 
%     for i = 1:30
%         csiamp_interp_wden_ant1(i,:) = wden(interp_data_ant1(i,:),'sqtwolog','s',scal,4,'sym3');
%         csiamp_interp_wden_ant2(i,:) = wden(interp_data_ant2(i,:),'sqtwolog','s',scal,4,'sym3');
%         csiamp_interp_wden_ant3(i,:) = wden(interp_data_ant3(i,:),'sqtwolog','s',scal,4,'sym3');
%     end
% 
%     %% PCA    
% 
%     dsc_ant1 = csiamp_interp_wden_ant1; %  dsc = 30 x 10000
%     dsc_ant2 = csiamp_interp_wden_ant2;
%     dsc_ant3 = csiamp_interp_wden_ant3;
%     
%     [PCs_ant1,~,eigenvalue_ant1] = PCA(dsc_ant1(1:30,:)',30); % PCs=10000x30, eigenvalue=30x1 30개로 분해된 PC와 그에 해당하는 eigenvalue
%     [PCs_ant2,~,eigenvalue_ant2] = PCA(dsc_ant2(1:30,:)',30); 
%     [PCs_ant3,~,eigenvalue_ant3] = PCA(dsc_ant3(1:30,:)',30); 
% 
%     % num of PCs = 10
%     csiamp_interp_wden_pca_ant1 = PCs_ant1(:, 1:10);
%     csiamp_interp_wden_pca_ant2 = PCs_ant2(:, 1:10);
%     csiamp_interp_wden_pca_ant3 = PCs_ant3(:, 1:10);
% 
%     result = cat(3, csiamp_interp_wden_pca_ant1, csiamp_interp_wden_pca_ant2, csiamp_interp_wden_pca_ant3);
%     size(result)
% 
%     new_name = extractBefore(nonfall_train_interp_list(w).name, length(nonfall_train_interp_list(w).name)-10); %remove '.dat'
%     file_name = strcat(new_name, '_wden_pca');
%     save(fullfile(pwd, '/dataset/nonfall_train_wden_pca', file_name), 'result');
%     whos('-file',fullfile(pwd, '/dataset/nonfall_train_wden_pca', file_name))
%     %fall_data = permute(fall_data, [3, 1, 2]);
% end
% 
% for w=1:length(nonfall_test_interp_list)
% % for w=1:1
% 
%     nonfall_test_interp_list(w).name
%     interp_mat_file = load(nonfall_test_interp_list(w).name);
%     interp_data_ant1 = interp_mat_file.result(:,:,1); % 30 x 10000
%     interp_data_ant2 = interp_mat_file.result(:,:,2);
%     interp_data_ant3 = interp_mat_file.result(:,:,3);
% 
% % 
% %     x1=1:1:length(interp_data_ant1(10,:));
% %     y1=interp_data_ant1;
% %     figure
% %     plot(x1,y1)
%     
%     %% wavelet denoising
%     scal = 'sln';%'mln''one'  Use model assuming standard Gaussian white noise.
% 
%     for i = 1:30
%         csiamp_interp_wden_ant1(i,:) = wden(interp_data_ant1(i,:),'sqtwolog','s',scal,4,'sym3');
%         csiamp_interp_wden_ant2(i,:) = wden(interp_data_ant2(i,:),'sqtwolog','s',scal,4,'sym3');
%         csiamp_interp_wden_ant3(i,:) = wden(interp_data_ant3(i,:),'sqtwolog','s',scal,4,'sym3');
%     end
% 
%     %% PCA    
% 
%     dsc_ant1 = csiamp_interp_wden_ant1; %  dsc = 30 x 10000
%     dsc_ant2 = csiamp_interp_wden_ant2;
%     dsc_ant3 = csiamp_interp_wden_ant3;
%     
%     [PCs_ant1,~,eigenvalue_ant1] = PCA(dsc_ant1(1:30,:)',30); % PCs=10000x30, eigenvalue=30x1 30개로 분해된 PC와 그에 해당하는 eigenvalue
%     [PCs_ant2,~,eigenvalue_ant2] = PCA(dsc_ant2(1:30,:)',30); 
%     [PCs_ant3,~,eigenvalue_ant3] = PCA(dsc_ant3(1:30,:)',30); 
% 
%     % num of PCs = 10
%     csiamp_interp_wden_pca_ant1 = PCs_ant1(:, 1:10);
%     csiamp_interp_wden_pca_ant2 = PCs_ant2(:, 1:10);
%     csiamp_interp_wden_pca_ant3 = PCs_ant3(:, 1:10);
% 
%     result = cat(3, csiamp_interp_wden_pca_ant1, csiamp_interp_wden_pca_ant2, csiamp_interp_wden_pca_ant3);
%     size(result)
% 
%     new_name = extractBefore(nonfall_test_interp_list(w).name, length(nonfall_test_interp_list(w).name)-10); %remove '.dat'
%     file_name = strcat(new_name, '_wden_pca');
%     save(fullfile(pwd, '/dataset/nonfall_test_wden_pca', file_name), 'result');
%     whos('-file',fullfile(pwd, '/dataset/nonfall_test_wden_pca', file_name))
%     %fall_data = permute(fall_data, [3, 1, 2]);
% end
% 

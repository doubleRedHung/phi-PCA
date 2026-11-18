% Pick one label to represent the dataset.
% Add t_3 noise to n_o=round(op*n) randomly selected samples.
% Report results under varying noise levels (or outlier proportions).

clear all
close all

load mnist.mat;
trainX = double(trainX);
p = size(trainX, 2);

%% set parameters
basis_seq = 50;
epsilon = 10^(-8);  
mvt_dof = 3;  % outlier dof

label = 1; % choose label 1 or 2 as the dataset
noise_level_seq = 300; %[100:100:700];   
op_seq = [0.0:0.05:0.3]; %0.1;   % outlier proportion

%% data
n_label = length(unique(trainY));

% Organize training data into trainC
trainC = cell(1, n_label);  % 10 cells, one for each digit (0-9)
for digit = 0:(n_label-1)
    indices = find(trainY == digit);   
    trainC{digit + 1} = trainX(indices, :);  
end 
x_tmp = trainC{1, label + 1};  

n = size(x_tmp, 1);
m = floor(sqrt(n)); 

gap = n - floor(n/m)*m;
n_m = zeros(m,1);
for i=1:m
    if i<(m-gap+1)
        n_m(i) = floor(n/m);
    else
        n_m(i) = floor(n/m)+1;
    end
end


%%
for inl = 1:length(noise_level_seq)
    for iop =1:length(op_seq)

        noise_level = noise_level_seq(inl);
        op = op_seq(iop);
        %% outiers 
        if op>0
            n_o = round(op*n);
            ind_o = randsample(1:n, n_o, false);
            trainX_tmp = x_tmp;
            x_tmp(ind_o,:) = trainX_tmp(ind_o,:) + noise_level*mvtrnd(eye(p), mvt_dof, n_o)./(mvt_dof/(mvt_dof-2))^0.5;    
        end
        mean_x_tmp = mean(x_tmp,1);
        
        %% PCA
        [uu_pca, ss_pca] = svds(cov(x_tmp), p);
        U_pca = uu_pca(:,1:basis_seq);
        
        %% phi-PCA
        HM_avr = zeros(p,p);
        GM_avr = zeros(p,p);
        for kk=1:m
            count = sum(n_m(1:kk-1));
            no_pc = min([p, n_m(kk)-1]);
            cov_kk = cov(x_tmp(count+1:count+n_m(kk),:)); 
        
            [uu_temp, ss_temp] = svds(cov_kk, no_pc);
            cov_e_kk = uu_temp*ss_temp*uu_temp' + (epsilon*trace(ss_pca)/p)*eye(p);
        
            HM_avr = HM_avr + inv(cov_e_kk);
            GM_avr = GM_avr + logm(cov_e_kk);
           
        end
        [U_hm, ss_hm] = svds(pinv(HM_avr/m), basis_seq);  % HM
        [U_gm, ss_gm] = svds(expm(GM_avr/m), basis_seq);  % GM
        
       
        U_global = cat(3, U_pca, U_hm, U_gm);
        U_all{inl, iop} = U_global;
        

        %% test image reconstructions
        if inl==1 && iop==1
            fig = figure;
        end
        
        idx = find(testY == label);   
        % indx = randsample(idx, 1, false)    
        if label==1
            indx = 4213   % label=1
        else label==2
            indx = 1507   % label=2
        end
        
        
        irow = size(U_global,3);
        icol = max(length(noise_level_seq), length(op_seq)) +1;
        if length(noise_level_seq)>1
            tmp = inl;
        else
            tmp = iop;
        end
        for i=1:irow
            
            if tmp==1
                subplot(irow, icol, tmp+(i-1)*icol);
                image = double(testX(indx,:));
                y = testY(indx);
                image_recnst = image;
                image_recnst = reshape(image_recnst,[28,28]);
                imshow(uint8(image_recnst'));
                title('True', fontsize=12);
            end
               
            subplot(irow, icol, 1+tmp+(i-1)*icol);  
            image_recnst = (image-mean_x_tmp)*U_global(:,:,i)*U_global(:,:,i)';  
            image_recnst = reshape(image_recnst + mean_x_tmp,[28,28]);          
            imshow(uint8(image_recnst'));   

            if length(noise_level_seq)>1
                title(['\sigma_{out} = ', num2str(noise_level)], fontsize=13); 
            else
                title(['\pi = ', num2str(op)], fontsize=13);   
            end         
        end
        if inl==1 && iop==1 
            annotation('textbox', [0.02, 0.625, 0.1, 0.2], 'String', 'PCA', ...
                        'EdgeColor', 'none', 'FontSize', 14, 'HorizontalAlignment', 'right');
        
            annotation('textbox', [0.02, 0.325, 0.1, 0.2], 'String', 'HM-PCA', ...
                        'EdgeColor', 'none', 'FontSize', 14, 'HorizontalAlignment', 'right');
        
            annotation('textbox', [0.02, 0.03, 0.1, 0.2], 'String', 'GM-PCA', ...
                        'EdgeColor', 'none', 'FontSize', 14, 'HorizontalAlignment', 'right');
        end  
        sgtitle({['The case of c = ', num2str(label)]}, 'FontSize', 18);
        set(gcf, 'Position', [40, 160, 1900, 800]);  

    end
end


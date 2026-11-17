clear all
close all

load('mnist.mat');

n = 60000; 
m = 100; %floor(sqrt(n));
nt = 10000; 
NumCmp = 30;
contam_rate =1; % contamination rate

n_outs= floor(n*contam_rate); % no. outliers/contaminations
noise_level = 200;
mvt_dof =3;
BG = noise_level*mvtrnd(eye(28^2), mvt_dof, n_outs)./(mvt_dof/(mvt_dof-2))^0.5;
idx=randperm(n)<=n_outs;
x= double(trainX);
x(idx,:) = x(idx,:)+BG;

eig_true = pca_ppca_hmpca('pca', double(trainX'), NumCmp);
eig_hm = pca_ppca_hmpca('hmpca', x', NumCmp, m); 
eig_ppca = pca_ppca_hmpca('ppca', x', NumCmp); 
eig_pca = pca_ppca_hmpca('pca', x', NumCmp); 

sim_hm = mean(svd(eig_true'*eig_hm));
sim_ppca = mean(svd(eig_true'*eig_ppca));
sim_pca = mean(svd(eig_true'*eig_pca));

y=double(trainY);
c_hm = classify(double(testX)*eig_hm,x*eig_hm,y,'quadratic');
c_ppca = classify(double(testX)*eig_ppca,x*eig_ppca,y,'quadratic');
c_pca = classify(double(testX)*eig_pca,x*eig_pca,y,'quadratic');
disp('baseline accuracy by QDA using 30 PCs = 0.96')
acc_hm =mean(c_hm'==double(testY));
acc_ppca =mean(c_ppca'==double(testY));
acc_pca =mean(c_pca'==double(testY));
acc = [acc_hm,acc_ppca,acc_pca]
subspace_similarity = [sim_hm,sim_ppca,sim_pca]



%{\
% plotting reconstructions
for irep =1:20
tmp = find(idx);
ii= randsample(1:n_outs,1); ii = tmp(ii);
plot_slide = floor(NumCmp/5);
figure
for j=1:5
    subplot(3,6,j)
    tmp = eig_hm(:,1:plot_slide*j);
    x_recnst = reshape(tmp*(tmp'*x(ii,:)'),[28,28]);
    imshow(uint8(x_recnst'));
    title(['HM (',num2str(plot_slide*j),')'])
end
subplot(3,6,6)
ground_truth=reshape(x(ii,:)',[28,28]);
imshow(uint8(ground_truth'));
title('noisy image')

for j=1:5
    subplot(3,6,j+6)
    tmp = eig_ppca(:,1:plot_slide*j);
    x_recnst = reshape(tmp*(tmp'*x(ii,:)'),[28,28]);
    imshow(uint8(x_recnst'));
    title(['PPCA (',num2str(plot_slide*j),')'])
end
subplot(3,6,12)
ground_truth=reshape(x(ii,:)',[28,28]);
imshow(uint8(ground_truth'));
title('noisy image')

for j=1:5
    subplot(3,6,j+12)
    tmp = eig_pca(:,1:plot_slide*j);
    x_recnst = reshape(tmp*(tmp'*x(ii,:)'),[28,28]);
    imshow(uint8(x_recnst'));
    title(['PCA (',num2str(plot_slide*j),')'])
end
subplot(3,6,18)
ground_truth=reshape(x(ii,:)',[28,28]);
imshow(uint8(ground_truth'));
title('noisy image')
end
%}




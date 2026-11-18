close all
clear variables

irep = 50;
n = 400;   % 1000
p = 1000;  % 200
r = 10;   % true rank
ru = 50;  %  upper bound of r

m = floor(n^0.5); % no. partition
% m = 3;

epsilon = 10^-8;   % -5

%%
rnd_Gamma = 1;   % random basis
theta = 0.5;   % noise eigenvalues ~ U(theta, 2-theta)
% mvt_ind = 2;   % mvt: CPCA >> PCA
% dof = 3;   % for data, 2.5 makes CPCA better in estimating eigen-space

op_seq = [0.05, 0.1];   % outlier proportion, 0.2
o_dof = 1; % outlier dof

o_mean_size = 0;   % outlier mean size, 0
o_std_seq = [1, 1000];   % --> affect no. distinct outluers (random directions)

o_sig_ind = 1;   % outlier at signal eigenvectors
o_e = p;  % no. eigenvectors for outlier (critical!) -> space for outliers
% o_r <= 3 (without signal): HM > beta=0
% outlier directions ~ min(n*op, o_r)
% sparse(small o_e) + inner(small o_std_size) + pure(o_sig_ind=0): PPCA>beta>HM

%%
Lambda_signal = 1 + (p/n)^0.5 + p.^(1./[2:(3/(r-1)):5])';
% Lambda_signal = 1 + (p/n)^0.5 + p.^(1./[2:r+1])';


Lambda = [Lambda_signal; unifrnd(theta, 2-theta, p-length(Lambda_signal),1)];

if rnd_Gamma == 1
    [Gamma, ~] = qr(normrnd(0,1,p,p));
else
    Gamma = eye(p);
end
Sigma = Gamma*diag(Lambda)*Gamma';
Sigma_half = Gamma*diag(Lambda.^(0.5))*Gamma';


%%
for i_op = 1:length(op_seq)
for i_o = 1:length(o_std_seq)
for i = 1:irep
op = op_seq(i_op)
o_std = o_std_seq(i_o)
% o_mean_size = o_std_seq(i_o)

% if mvt_ind == 1
%     x = mvtrnd(eye(p), dof, n)*Sigma_half./(dof/(dof-2))^0.5;   % CPCA > PCA
% else
%     % x = trnd(dof, n,p)*Sigma_half./(dof/(dof-2))^0.5; 
%    x = mvnrnd(zeros(p,1), Sigma, n); 
% end
x = mvnrnd(normrnd(0,0,p,1), Sigma, n);
[vv_pca0, ss_pca0] = svds(cov(x), p);  %%% optimal PCA

%%% outliers %%%
n_o = round(n*op);
mu_o = normrnd(0,1,p,1);
mu_o = o_mean_size * mu_o/norm(mu_o);
if o_sig_ind == 1   % with signal
    % ind_temp = [1:r, randsample(r+1:p, o_e-r)];
    ind_temp = [1:p];
else
    ind_temp = randsample(r+1:p, min([o_e, p-r]));   % without signal 
end
oP = Gamma(:,ind_temp)*Gamma(:,ind_temp)';
% x_o = (repmat(mu_o',n_o,1)+o_std*(mvtrnd(eye(p),o_dof,n_o))./(o_dof/(o_dof-2))^0.5)*oP;
x_o = (repmat(mu_o',n_o,1)+o_std*(mvtrnd(eye(p),o_dof,n_o)))*oP;



x(1:n_o,:) = x_o;
x = x(randperm(n),:);


%%

%%% PCA %%%
[vv_pca, ss_pca] = svds(cov(x), p);

%%% PPCA %%%
n1 = round(n/2);
M1 = cov(x(1:n1,:));
[u1, s1] = svd(M1);
M1_half = u1*diag(diag(s1.^0.5))*u1';
M2 = cov(x(n1+1:end,:));
[u2, s2] = svd(M2);
M2_half = u2*diag(diag(s2.^0.5))*u2';

[uu_ppca, ss_ppca, vv_ppca] = svds(M1_half*M2_half, ru);
uu_c_ppca = [];
for jj = 1:r-1
    [uu_c_temp, ss_c] = svds([uu_ppca(:,jj),vv_ppca(:,jj)], 1);
    uu_c_ppca(:,jj) = uu_c_temp;
end

%%% GM2 %%%
GM2 = expm((logm(M1+(epsilon*trace(ss_pca)/p)*eye(p)) + logm(M2+(epsilon*trace(ss_pca)/p)*eye(p)))/2);
[vv_gm2, ss_gm2] = svds(GM2, ru);


%%% phi-PCA %%%
HM_avr = zeros(p,p);
GM_avr = zeros(p,p);
AM_avr = zeros(p,p);
n_m = floor(n/m); % for RP
rr = n - m*n_m;
part_ind_1 = reshape(1:(n_m+1)*rr, n_m+1, rr);   % size = n_m+1
part_ind_2 = reshape((n_m+1)*rr+1:n, n_m, m-rr);   % size = n_m
for kk=1:m
    if kk <= rr
        part_ind = part_ind_1(:,kk);        
    else
        part_ind = part_ind_2(:,kk-rr);
    end
    % no_pc = min([p, length(part_ind)-1]);
    cov_kk = cov(x(part_ind,:));
    % [uu_temp, ss_temp] = svds(cov_kk, no_pc);
    % cov_e_kk = uu_temp*ss_temp*uu_temp' + (epsilon*trace(ss_pca)/p)*eye(p);
    cov_e_kk = cov_kk + (epsilon*trace(ss_pca)/p)*eye(p);

    HM_avr = HM_avr + pinv(cov_e_kk);
    GM_avr = GM_avr + logm(cov_e_kk);
    AM_avr = AM_avr + cov_kk;

    % HM_avr = HM_avr + inv(cov_kk + (epsilon*trace(ss_pca)/p)*eye(p));
    % GM_avr = GM_avr + uu_temp*diag(log(diag(ss_temp)))*uu_temp';
end
[vv_hm, ss_hm] = svds(pinv(HM_avr/m), ru);  % HM
[vv_gm, ss_gm] = svds(expm(GM_avr/m), ru);  % GM
[vv_am, ss_am] = svds(AM_avr/m, ru);  % AM

%%
for jj = r:ru
    mean_similarity(i,i_op,i_o,1,jj) = mean(svds(vv_hm(:,1:jj)'*Gamma(:,1:r), r));  % HM
    mean_similarity(i,i_op,i_o,2,jj) = mean(svds(vv_gm(:,1:jj)'*Gamma(:,1:r), r));  % GM
    mean_similarity(i,i_op,i_o,3,jj) = mean(svds(vv_am(:,1:jj)'*Gamma(:,1:r), r));  % AM
    mean_similarity(i,i_op,i_o,4,jj) = mean(svds(vv_pca(:,1:jj)'*Gamma(:,1:r), r));  % PCA
    mean_similarity(i,i_op,i_o,5,jj) = mean(svds(vv_pca0(:,1:jj)'*Gamma(:,1:r), r));  % opt-PCA
    mean_similarity(i,i_op,i_o,6,jj) = mean(svds(vv_gm2(:,1:jj)'*Gamma(:,1:r), r));  % GM2

    [uu_c_temp, ss_temp] = svds([uu_ppca(:,jj),vv_ppca(:,jj)], 1);
    uu_c_ppca(:,jj) = uu_c_temp;
    [uu_c_ppca_qr, R] = qr(uu_c_ppca(:,1:jj), 0);
    mean_similarity(i,i_op,i_o,7,jj) = mean(svds(uu_c_ppca_qr'*Gamma(:,1:r), r));  % PPCA
    
end
end
end
end
mean_similarity = squeeze(mean(mean_similarity,1));


%%

figure
for i_o = 1:length(o_std_seq)
for i_op = 1:length(op_seq)

subplot(length(o_std_seq), length(op_seq), (i_o-1)*length(o_std_seq)+i_op)
hold on
box on

h2 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 2, r:ru)), '-ko');  % GM
h1 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 1, r:ru)), '-r.');  % HM

h3 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 3, r:ru)), ':bs');  % AM
h4 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 4, r:ru)), ':b.'); % PCA
h5 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 5, r:ru)), ':b');  % opt-PCA

h6 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 6, r:ru)), '-k|');  % GM2
h7 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 7, r:ru)), '--r|');  % PPCA  

if i_o+i_op == 2
% legend([h1,h2,h3,h4,h5], 'HM-PCA','GM-PCA','AM-PCA','PCA','opt-PCA','Location', 'SouthEast')
legend([h1,h2,h3,h6,h7,h4,h5], 'HM-PCA','GM-PCA','AM-PCA','GM-PCA2','PPCA','PCA','opt-PCA','Location', 'SouthEast')
end

xlim([r-1, ru+1])
ylim([0.0, 1.05])
title(['(\pi, \sigma_{out}) = (',num2str(op_seq(i_op)),', ',num2str(o_std_seq(i_o)),')'])
ylabel('s_q')
xlabel('q')
end
end


%%
%{
for ii = 1: n
    d_s = sum((x(ii,:)*Gamma(:, 1:r)).^2./Lambda(1:r)');
    d_n = sum((x(ii,:)*Gamma(:, r+1:end)).^2./Lambda(r+1:end)');
    Dx(ii) = d_n/(p-r)-d_s/r;   % Delta(x)
    md(ii) = d_s + d_n;
end

subplot(1,2,1) 
hold on
box on
h1= plot(1:n, Dx, '-r.');
legend(h1, '\Delta(x)')

subplot(1,2,2) 
hold on
box on
h1= plot(1:n, md, '-r.');
legend(h1, 'M dist.')
%}

%%
%{
for ii = 1: n_o
    x_s = sum((x_o(ii,:)*Gamma(:, 1:r)).^2./Lambda(1:r)');
    x_n = sum(((x_o(ii,:)*Gamma(:, r+1:end))).^2./Lambda(r+1:end)');
    delta(ii,1) = (x_s+x_n-1)*(x_n/(p-r)-x_s/r);  
    delta(ii,2) = p*(x_s/(p-r))*(x_n/r);
end

figure 
hold on
box on

h1= plot(1:n_o, delta(:,1)./delta(:,2), '-r.');
h2= yline(1, 'k:');

legend(h1, 'ratio')
%}



%%
%{
figure
for i_o = 1:length(o_std_seq)
for i_op = 1:length(op_seq)

subplot(length(o_std_seq), length(op_seq), (i_o-1)*length(o_std_seq)+i_op)
hold on
box on

h2 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 2, r:ru)), '-k.');  % GM
h6 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 6, r:ru)), '-ko');  % GM2
h7 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 7, r:ru)), '-r|');  % PPCA    
h4 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 4, r:ru)), '-bs'); % PCA
h5 = plot([r:ru], squeeze(mean_similarity(i_op, i_o, 5, r:ru)), ':b');  % opt-PCA

legend([h2,h6,h7,h4,h5], 'GM-PCA','GM-PCA2','PPCA','PCA','opt-PCA','Location', 'SouthEast')   
xlim([r-1, ru+1])
ylim([0.0, 1.05])
title(['(\pi, \sigma_{out}) = (',num2str(op_seq(i_op)),', ',num2str(o_std_seq(i_o)),')'])
ylabel('s_q')
xlabel('q')
end
end
%}
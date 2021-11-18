function [pca_data, mda_data, y] = loadandfiddle()
%% Load DATA here

face = load('data.mat').face;

neutral_inds = 3*(1:200) - 2; % Indices corresponding to neutral faces
neutral_faces = face(:,:, neutral_inds); % Neutral faces
neutral_faces_flattened = reshape(neutral_faces, 24*21, 200); % Neutral face data flattened 
neutral_faces_flattened = neutral_faces_flattened' ; % Rows represent obs.

% Do the same for faces w expression 
express_inds = 3*(1:200) - 1; 
express_faces = face(:,:, express_inds); 
express_faces_flattened = reshape(express_faces, 24*21, 200); 
express_faces_flattened = express_faces_flattened'; 

classes{1} = neutral_faces_flattened; 
classes{2} = express_faces_flattened; 

%% Fiddle here
recentered_classes = {}; % store recentered data
covariances = {}; % store covariances
means = {}; % store centers of all data
for i = 1:2
    avg = (1/size(classes{i},1))*ones(1,size(classes{i},1))*classes{i}; 
    recentered_classes{i} = classes{i} - ...
                            repmat(avg, size(classes{i},1),1); % recenter
    means{i} = avg; % compute class mean 
    covariances{i} = (1/size(classes{i},1))* ...
                     recentered_classes{i}' * recentered_classes{i};
    % compute covariance
end

data = [];
recentered_data = []; 
%y = []; 

for i = 1:2
    data = [data; classes{i}];
    recentered_data = [recentered_data; recentered_classes{i}];
    %y = [y; i*ones(size(classes{i},1),1)];
end

avg = (1/size(data,1))*ones(1,size(data,1))*data; 

Sigma_b = zeros(size(covariances{1})); % between class scatter 
for i=1:2
    Sigma_b = Sigma_b + (size(classes{i},1)/size(data,1))* ...
              (means{i} - avg)' * (means{i} - avg); 
end

Sigma_w = zeros(size(covariances{1})); % within class scatter
for i = 1:2
    Sigma_w = Sigma_w + (size(classes{i},1)/size(data,1))*covariances{i};
end


%% prep for pca
centered_data = data - repmat(avg, size(data,1), 1); 
covar = centered_data' * centered_data; 
 
%%  eigenvalue computation here
[V,D] = eigs(covar,10); 
pca_data = data*real(V); 

%% MDA here
Sigma_w = Sigma_w + eye(size(Sigma_w)); 
[A, Sigmas] = eigs(Sigma_b, Sigma_w,15); 
mda_data = data*real(A);
%% analyse the eigenvalues 

% pca: 
mult_pca = triu(ones(size(D))); 
eig_sums_pca = mult_pca' * diag(D);
perc_eig_sums_pca = diag(D(2:end, 2:end))./eig_sums_pca(2:end);
ratios_pca = diag(D(2:end,2:end))./diag(D(1:end-1, 1:end-1));

% mda: 
mult_mda = triu(ones(size(Sigmas))); 
eig_sums_mda = mult_mda' * diag(Sigmas);
perc_eig_sums_mda = diag(Sigmas(2:end, 2:end))./eig_sums_mda(2:end);
ratios_mda = diag(Sigmas(2:end,2:end))./diag(Sigmas(1:end-1, 1:end-1));

%% plot

figure(); 
plot(perc_eig_sums_pca, 'bo-');
%hold on;
%plot(perc_eig_sums_mda, 'ro-'); 
%hold on; 
%plot(0.05*ones(size(perc_eig_sums_mda)), 'k-');
title('Percentage change in running eigenvalue sum for PCA eigenvalues', 'Interpreter', 'latex'); 
xlabel('N', 'Interpreter', 'latex');
ylabel('$\lambda_N/\sum_{i=1}^{N-1}\lambda_i$','Interpreter', 'latex'); 
%legend('PCA', 'MDA', 'Interpreter', 'latex');
%% plot MDA eigenvalues here
% figure();
% plot(1:15, real(diag(Sigmas)), 'bo'); 
% title('Generalized eigenvalues for $\Sigma_b = \Sigma_w C D$ in the MDA problem', 'Interpreter', 'latex'); 
% xlabel('N', 'Interpreter', 'latex');
% ylabel('$\lambda_N$', 'Interpreter', 'latex'); 

%% 
% Full matrix construction

% Here each row represents an observation, the number of columns is 
% the number of pixels in each image 

Raw_data = [neutral_faces_flattened; express_faces_flattened]; 
y = [ones(200,1); -1*ones(200,1)]; 
% PCA it here
[coeff, score, latent] = pca(Raw_data); % pca this shit 
% coeff are the eigenvectors
% score are the projections
% latent are the eigenvalues 
pca_data = score(:,1:50); % Pick the first 50 principal components based on 
                      % eigenvalue decay 
mda_data = mda_data(:,[1;2;10]); 
% Note: Score coordinates 2,3,4 lead to highly linearly separable data. 
% Use this! 
%% Write data to file
end



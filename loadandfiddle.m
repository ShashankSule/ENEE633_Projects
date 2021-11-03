%% Load DATA here

load('data.mat')

neutral_inds = 3*(1:200) - 2; % Indices corresponding to neutral faces
neutral_faces = face(:,:, neutral_inds); % Neutral faces
neutral_faces_flattened = reshape(neutral_faces, 24*21, 200); % Neutral face data flattened 
neutral_faces_flattened = neutral_faces_flattened' ; % Rows represent obs.

% Do the same for faces w expression 
express_inds = 3*(1:200) - 1; 
express_faces = face(:,:, express_inds); 
express_faces_flattened = reshape(express_faces, 24*21, 200); 
express_faces_flattened = express_faces_flattened'; 

% Full matrix construction

% Here each row represents an observation, the number of columns is 
% the number of pixels in each image 

Raw_data = [neutral_faces_flattened; express_faces_flattened]; 
labels = [ones(200,1); 2*ones(200,1)]; 
%% PCA it here
[coeff, score, latent] = pca(Raw_data); % pca this shit 
% coeff are the eigenvectors
% score are the projections
% latent are the eigenvalues 
data = score(:,1:50); % Pick the first 50 principal components based on 
                      % eigenvalue decay 

%% Write data to file





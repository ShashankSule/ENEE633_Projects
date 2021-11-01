%% Load DATA here

load('data.mat')

neutral_inds = 3*(1:200); % Indices corresponding to neutral faces
neutral_faces = face(:,:, neutral_inds); % Neutral faces
neutral_faces_flattened = reshape(neutral_faces, 24*21, 200);
neutral_faces_flattened = neutral_faces_flattened' ; % Rows represent obs.
% Neutral face data flattened 




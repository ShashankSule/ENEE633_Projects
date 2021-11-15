%% we'll write a bayes classifier for mda-illuminated data

training = mda_data; 
% y = y are the labels
test = training(400,:);
%% computing class covariances etc 
classes = {}; % store class data
recentered_classes = {}; % store recentered data
covariances = {}; % store covariances 
means = {}; % store means 
for i = 1:68
    classes{i} = training(y == i, :);
    means{i} = (1/size(classes{i},1))*ones(1,size(classes{i},1))*classes{i}; % compute mean
    recentered_classes{i} = classes{i} - ...
                            repmat(means{i}, size(classes{i},1),1); % recenter
    covariances{i} = (1/size(classes{i},1)) * ...
                      recentered_classes{i}' * recentered_classes{i}; % covariances
end

%% now do the classification 

% first compute expression of the form Sigma_i^-1 * (x - mu_i) 
K = size(classes,2); % # of classes
z = zeros(size(means{1},2),K); % store Sigma_i^-1 * (x - mu_i) 
for i = 1:K 
    z(:,i) = covariances{i} \ (test - means{i})';
end
dets = cellfun(@det, covariances); % store all the determinants
class_vec = zeros(K,1); % store the values G_j exp((x-mu_j) * z_j)
for i = 1:K
    class_vec(i) = (1/dets(i)) * exp(-0.5 * (test - means{i}) * z(:,i));
end

label = find(class_vec == max(class_vec), 1); 

%% we'll write a bayes classifier for mda-illuminated data

[pca_data,~,y] = loadandfiddleIllumination();
testing_err_Bayes = zeros(19,1);
%%
for K = 2:20
fprintf('Training classifiers for %d points per class \n', K); 
% Get training data and testing data
n_classes = size(unique(y),1); 
size_classes = sum(y==1);
%K = 3; % number of training points in each class 
training_inds = repelem(size_classes*(0:(n_classes-1)) + 1, K) + repmat(0:(K-1), 1, 68);
training = pca_data(training_inds, :); 
training_labels = y(training_inds);
test_inds = setdiff(1:size(pca_data,1),training_inds);
test = pca_data(test_inds,:);
test_labels = y(test_inds); 

%training = mda_data; 
% y = y are the labels
%test = tr
% computing class covariances etc 
classes = {}; % store class data
recentered_classes = {}; % store recentered data
covariances = {}; % store covariances 
means = {}; % store means 
log_dets = zeros(n_classes,1); % store log of determinants 
for i = 1:n_classes
    classes{i} = training(training_labels == i, :);
    means{i} = (1/size(classes{i},1))*ones(1,size(classes{i},1))*classes{i}; % compute mean
    recentered_classes{i} = classes{i} - ...
                            repmat(means{i}, size(classes{i},1),1); % recenter
    covariances{i} = (1/size(classes{i},1)) * ...
                      recentered_classes{i}' * recentered_classes{i}; % covariances
    e_vals = eig(covariances{i}); 
    log_dets(i) = sum(log(e_vals(e_vals > 1e-10))); % determinants of covariances 
end
fprintf('Trained! Now testing ...\n');
% now do the classification 

% first compute expression of the form Sigma_i^-1 * (x - mu_i)
label = zeros(size(test,1),1);
for j=1:size(test,1)
    %K = size(classes,2); % # of classes
    z = zeros(size(means{1},2),n_classes); % store Sigma_i^-1 * (x - mu_i) 
    for i = 1:n_classes 
        %z(:,i) = pinv(covariances{i}) * (test(j,:) - means{i})';
        z(:,i) = lsqminnorm(covariances{i}, (test(j,:) - means{i})');
    end
    class_vec = zeros(n_classes,1); % store the values G_j exp((x-mu_j) * z_j)
    for i = 1:n_classes
        class_vec(i) = -0.5*size(covariances{1},1)*log_dets(i) -0.5*(test(j,:) - means{i})*z(:,i);
        % compute log likelihood
    end

    label(j) = find(class_vec == max(class_vec), 1); 
end

% compute testing error
testing_err_Bayes(K,1) = mean(test_labels ~= label); 
end



%% plot error data
plot(2:20, 100*testing_err_Bayes(2:end), 'bo-');
xlabel("Number of training points per class", 'Interpreter', 'latex'); 
ylabel("Percentage testing error", 'Interpreter', 'latex'); 
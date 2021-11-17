%% Let's test the bayes' classifier here on DATA

%% Set up data
[mda_data, ~, y] = loadandfiddle();
props = linspace(0.1,0.9,15); 
testing_err = zeros(15,1); 

%% Now test! 
for l = 1:15
prop = props(l); 
fprintf("Computing bayes labels using %d proportion of data from each class\n", prop); 
% train
n_classes = size(unique(y),1); 
size_classes = sum(y==1); 
K = floor(prop*size_classes); % number of training points from each class 
training_inds = repelem(size_classes*(0:(n_classes-1)) + 1, K) + repmat(0:(K-1), 1, 2);
training = mda_data(training_inds, :); 
training_labels = y(training_inds);
test_inds = setdiff(1:size(mda_data,1),training_inds);
test = mda_data(test_inds,:);
test_labels = y(test_inds); 

% compute class parameters 
n = size(training,1);
[class1, class2, mu_1, mu_2, sigma_1, sigma_2] = preprocess(training,training_labels); 
bayes_labels = zeros(size(test,1),1); 
% now classify 
for i=1:size(bayes_labels)
    bayes_labels(i) = BayesClassifier_DATA(test(i,:),class1, class2, ...
                                      mu_1, mu_2, sigma_1, sigma_2); 
end

% Compute testing error 

testing_err(l) = mean(bayes_labels ~= test_labels); 

end

%% plot 
figure(); 
for l = 1:15 
    plot(sigma_params, 100*cval_errors(l,:), 'o-', 'DisplayName', ...
         num2str(floor(props(l)*size_classes)));
    xlabel("$\sigma$", 'Interpreter', 'latex'); 
    ylabel("Average cross-validation percentage error", 'Interpreter', 'latex'); 
    set(gca, 'FontSize', 12); 
    hold on; 
end
%% plot optimal sigmas

opt_sigmas = zeros(length(props),1); 
for i = 1:length(props)
    [~,ind] = min(cval_errors(i,:));
    opt_sigmas(i) = sigma_params(ind); 
end

plot(floor(props*200),opt_sigmas, 'bo-');
xlabel("Number of training points per class", 'Interpreter', 'latex'); 
ylabel("Cross-validation error minimimzing variance parameter", 'Interpreter', 'latex'); 

%% svm classification 

[mda_data,~, y] = loadandfiddle();
props = linspace(0.1,0.9,15);
testing_err = zeros(length(props),1); 
%boosting_err = zeros(5,length(props)); 
%opt_r obtained from crossval 
%% Do the testing 
%for K = 1:5 
    %fprintf("Computing %d Boosts...\n",2*K); 
    for l = 1:length(props)
        % set up training and test
        %l = 3; 
        prop = props(l); 
        fprintf("Training using %d proportion of data from each class\n", prop); 
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
        %r = 2; 
        sigma = 100;
        %k = @(x,y) exp(-((norm(x-y))^2)/(2*sigma)); 
        %k = @(x,y) (1 + x*y')^r;
        k = @(x,y) x*y'; 
        %labels = svmkernel(training, training_labels, test, k); 
        [labels,thetas,B,A,P] = adaBoost(training, training_labels, test,5, k);
        testing_err(l) = mean(test_labels ~= labels); 
    end
%end
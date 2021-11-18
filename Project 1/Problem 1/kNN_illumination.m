%% let's do a supervised k-nn here! 
[pca_data,~,y] = loadandfiddleIllumination(); 

%% get training and test data
testing_err = zeros(20,20);
for kk = 1:20
    for K = 1:20
    fprintf('Training for %d nearest neighbours on %d points per class...\n', kk, K); 
    n_classes = size(unique(y),1); 
    size_classes = sum(y==1);
    %K = 3; % number of training points in each class 
    training_inds = repelem(size_classes*(0:(n_classes-1)) + 1, K) + repmat(0:(K-1), 1, 68);
    training = pca_data(training_inds, :); 
    training_labels = y(training_inds);
    test_inds = setdiff(1:size(pca_data,1),training_inds);
    test = pca_data(test_inds,:);
    test_labels = y(test_inds); 

    % the actual classifier

    %k = K-1; 
    label = zeros(size(test,1),1); 
    for j=1:size(test,1)
        distances = zeros(size(training,1),1); %vector of distances

        for i = 1:size(training,1)
            distances(i) = norm(test(j,:) - training(i,:));
        end

        % sort the vector in ascending order 
        [~,inds] = sort(distances, 'ascend'); 
        k_labels = training_labels(inds(1:kk)); %find k nearest labels 
        A = tabulate(k_labels); 
        label(j) = find(A(:,3) == max(A(:,3)),1); % find the most frequent 
                                                      % label among k-NN's
    end
    % compute training error

    testing_err(kk,K) = mean(test_labels ~= label); 
    end
end

%% plot 
set(0,'defaulttextinterpreter','latex')
imagesc(testing_err); 
xlabel("Number of training points per class", 'Interpreter', 'latex'); 
ylabel("Number of nearest neighbours ($k$)", 'Interpreter', 'latex'); 
colormap copper; 
c = colorbar;
c.Label.String = "Training error";

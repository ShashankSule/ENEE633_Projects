%% cross-validation here

%% set up data

[~,mda_data, y] = loadandfiddle();
big_params = (10*(1:15))';
sigma_params = big_params;
props = linspace(0.1,0.9,15);
cval_errors = zeros(length(props), length(sigma_params));
M = length(sigma_params); 
%% compute cross validation!! 
parfor l = 1:length(props)
    %subroutine for selecting training data
    %l = 15; 
    prop = props(l); 
    fprintf("Cross validating using %d proportion of data from each class\n", prop); 
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
    parameters = sigma_params;
    % subroutine for cross validation
    for i = 1:M
    fprintf("Sigma = %d \n", parameters(i)); 
    sigma = parameters(i); % variance parameter 
    k = @(x,y) exp(-(norm(x-y)^2)/(2*sigma));
    %k = @(x,y) (1 + x*y')^(sigma);
    cval_errors(l,i) = crossval('mcr', training, training_labels, 'Predfun',...
                    @(xtrain, ytrain, xtest)svmkernel(xtrain, ytrain, xtest, k), ...
                    'KFold', 5);
    end
end
%% plot overall data
for i=1:length(props)
    plot(sigma_params, 100*cval_errors(i,:), 'o-', 'DisplayName', num2str(props(i)*200)); 
    xlabel("r", 'Interpreter', 'latex');
    ylabel("Average cross-validation percentage error", 'Interpreter', 'latex'); 
    title("Error statistics for tuning $r$ in polynomial kernel via 5-fold cross-validation", 'Interpreter', 'latex');
    hold on
end
%% plot the error data
opt_sigma = zeros(length(props),1); 
big_params = (10*(1:15))';
sigma_params = [0.5; big_params];
for i = 1:length(props) 
    [~,ind] = min(cval_errors(i,:)); 
    opt_sigma(i) = sigma_params(ind); 
end

plot(floor(props*200), opt_r, 'bo-'); 
xlabel("Number of training points per class", 'Interpreter', 'latex'); 
ylabel("Cross-validation error minimizing $r$", 'Interpreter', 'latex'); 

%% plot the error data as heatmap 
props = linspace(0.1,0.9,15);
imagesc(100*cval_errors(:,2:end)); colormap copper; 
xlabel("$\sigma$", 'Interpreter','latex');
ylabel("Proportion of training points per class", 'Interpreter', 'latex'); 
c = colorbar; 
c.Label.String = "Cross-validation error";
xt = get(gca, 'XTick'); 
xtlbl = sigma_params; 
set(gca, 'XTick',xt, 'XTickLabel',xtlbl, 'XTickLabelRotation',30)
yt = get(gca, 'YTick'); 
ytlbl = props(2*(1:7)); 
set(gca, 'YTick',yt, 'YTickLabel',ytlbl, 'XTickLabelRotation',0)
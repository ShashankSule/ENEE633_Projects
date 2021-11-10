%% let's do AdaBoost here! 
% First a set up  + initializations
K = 10; % Number of boosts! 
[data, y] = loadandfiddle(); % Load data and labels 
training = data; % set training set
n = size(y,1); % number of data points
weights = (1/n)*ones(n,1);% initialize every point with the same weight
thetas = []; % K x size(training,2) vector to store all the classifiers 
B = []; % store all the constant shift terms (we'll add it later to the classifiers) 
A = []; % K x 1 vector to store all the weights 
P = []; 
sigma = 0.1; % kernel bandwidth
k = @(x,y) exp(-(norm(x-y)^2)/(2*sigma)) ; %kernel 
% form kernel matrix: we'll need this for evaluation 
for i = 1:n
    for j = 1:n
        XX(i,j) = k(training(i,:), training(j,:)); % calculate k(x_i, x_j) 
    end
end

%% boosting algorithm 
for i=1:K 
    fprintf('Boosting... \n'); 
    p = weights/sum(weights); % Renormalized weight vectors
    [theta,b,epsilon] = WeakClassifier(p, training, y, k); % Module to compute a weak classifier
    F = (XX')*theta + b; % F, aka the classifier values 
    a = 0.5*log((1 - epsilon)/epsilon); % Compute weight 
    weights = weights.*exp(-y.*F); % wn_i+1 = wn_i * exp(-y_n * F(x_n))
    thetas = [thetas; theta']; % put the weak classifier vectors in 
    B = [B; b]; % update the constant terms in the classifier
    A = [A; a]; % update the weight on each classifier
    P = [P p]; % keep track of the weights for diagnostic purposes
end

% Finally, to compute the value of the boosted classifier F at x we do 
% F(x) = A' * thetas * [K(x,x_i)]_i + B 


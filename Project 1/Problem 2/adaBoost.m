function [label, thetas, B, A, P] = adaBoost(training, y, test, K, k) 
% arguments:
% training -- training data as n x d matrix
% y -- training labels as n x 1 matrix 
% x -- test object as M x d matrix
% K -- number of boosts integer
% k -- kernel: R^d x R^d --> R 
%% let's do AdaBoost here! 
% First a set up  + initializations
% K = 10; % Number of boosts! 
%[data, y] = loadandfiddle(); % Load data and labels 
%training = data; % set training set
n = size(y,1); % number of data points
weights = (1/n)*ones(n,1);% initialize every point with the same weight
thetas = []; % K x size(training,2) vector to store all the classifiers 
B = []; % store all the constant shift terms (we'll add it later to the classifiers) 
A = []; % K x 1 vector to store all the weights 
P = []; 

if ~exist('k', 'var')
    sigma = 0.1; % kernel bandwidth
    k = @(x,y) exp(-(norm(x-y)^2)/(2*sigma)) ; %kernel
end

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
    [theta,b,epsilon] = WeakClassifier(p, training, y, k, 0.1); % Module to compute a weak classifier
    F = (XX')*theta + b; % F, aka the classifier values 
    a = 0.5*log((1 - epsilon)/epsilon); % Compute weight 
    weights = weights.*exp(-y.*F); % wn_i+1 = wn_i * exp(-y_n * F(x_n))
    %fprintf('Pausing...\n');
    %pause(3);
    thetas = [thetas; theta']; % put the weak classifier vectors in 
    B = [B; b]; % update the constant terms in the classifier
    A = [A; a]; % update the weight on each classifier
    P = [P p]; % keep track of the weights for diagnostic purposes
end

%% Finally, to compute the value of the boosted classifier F at x we do 
%test = data(1:end,:); 
M = size(test,1); % number of test points 
Kk = zeros(n,M);
for j = 1:M
    for i = 1:n 
        Kk(i,j) = k(test(j,:),training(i,:));
    end
end

label = sign(A' * ((thetas * Kk) + repmat(B, 1, M)))';
end
% F(x) = A' * thetas * [K(x,x_i)]_i + B 


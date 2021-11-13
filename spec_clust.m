%% let's do spectral clustering here!! 
load('hw06-data1.mat');
data = X'; 
n = size(data,1); % number of points
k = @(x,y) exp((-(norm(x-y))^2)/10); % kernel here
for i = 1:n
    for j = 1:n 
        K(i,j) = k(data(i,:), data(j,:)); % set up kernel matrix 
    end
end

%% compute the fiedler vector: this is the expensive step
K(K < 1e-6) = 0; %sparsify 
D = K*ones(size(K,1),1); 
L = sparse(D - K);
D = sparse(diag(D)); 
[V,D] = eigs(L,D);
fiedler = V(:,2); 

%% compute the labels and plot! 
labels = fiedler > 0;
scatter(data(:,1), data(:,2), 30, labels);




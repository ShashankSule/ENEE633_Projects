%% cross-validation here

%% set up data

[~,training, y] = loadandfiddle();
sigma_params = [1; 1e1; 1e2; 1e3; 1e4]; 
%% compute cross vals! 
for i = 1:5
fprintf("Sigma = %d \n", sigma_params(i)); 
sigma = sigma_params(i); % variance parameter 
k = @(x,y) exp(-(norm(x-y)^2)/(2*sigma));
err = crossval('mcr', training, y, 'Predfun',...
                @(xtrain, ytrain, xtest)svmkernel(xtrain, ytrain, xtest, k));
end
%% cross-validation here

sigma = 100; % variance parameter 
k = @(x,y) exp(-(norm(x-y)^2)/(2*sigma));
[training, y] = loadandfiddle(); 
%myfunction = @(training, y, test) svmkernel(training, y, test, k);
%% check the function out here! 
err = crossval('mcr', training, y, 'Predfun',...
                @(xtrain, ytrain, xtest)svmkernel(xtrain, ytrain, xtest, k));

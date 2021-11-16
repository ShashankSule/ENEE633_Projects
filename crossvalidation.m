%% cross-validation here

%% set up data

[training,~, y] = loadandfiddle();
sigma_params = [1;10;20;30;40;50;60;70;80;90;100;110;120;130;140;150]; 
err = zeros(length(sigma_params),1);
%% compute cross vals! 
for i = 1:length(sigma_params)
Sigma = sigma_params(i);
fprintf("Sigma = %d \n", sigma_params(i)); 
sigma = sigma_params(i); % variance parameter 
k = @(x,y) exp(-(norm(x-y)^2)/(2*sigma));
err(i) = crossval('mcr', training, y, 'Predfun',...
                @(xtrain, ytrain, xtest)svmkernel(xtrain, ytrain, xtest, k), ...
                'KFold', 5);
end

%% plot the error data
plot(sigma_params, 100*param_err, 'bo-'); 
xlabel("$\sigma$", 'Interpreter', 'latex');
xticks(sigma_params);
ylabel("Average cross-validation percentage error ", 'Interpreter', 'latex');
title("Error statistics for tuning $\sigma$ in the RBF kernel via 5-fold cross-validation", 'Interpreter', 'latex');
set(gca, 'FontSize', 16);

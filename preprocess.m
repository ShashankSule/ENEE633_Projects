function [class1, class2, mu_1, mu_2, sigma_1, sigma_2] = preprocess(data,y)
%% use this function to generate the ML estimators of raw data given labels
% y
%extract the two classes 
%[data, y] = loadandfiddle(); 
class1 = data(y == 1, :); 
class2 = data(y == -1, :); 
n = size(y); % number of measurements
%x = data(400,:); % a testing point

% compute class means 
mu_1 = (1/size(class1,1))*ones(1,size(class1,1))*class1; 
mu_2 = (1/size(class2,1))*ones(1,size(class2,1))*class2; 

% compute class covariance matrix:

sigma_1 = zeros(size(data,2)); 
sigma_2 = zeros(size(data,2)); 

% class 1 
for i = 1:size(class1,1)
    sigma_1 = sigma_1 + (class1(i,:) - mu_1)'*(class1(i,:) - mu_1);
end

sigma_1 = (1/size(class1, 1))*sigma_1; 

% class 2
for i = 1:size(class2,1)
    sigma_2 = sigma_2 + (class2(i,:) - mu_2)'*(class2(i,:) - mu_2);
end

sigma_2 = (1/size(class2, 1))*sigma_2;
end


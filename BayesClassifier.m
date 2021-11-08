function label = BayesClassifier(x)
%% Let's do Bae's classification here. 

% Recall that we model each class as a Gaussian with ML parameters and 
% then do a bayes' classifier using i = argmax p(x | i)p(i) 

%extract the two classes 
[data, y] = loadandfiddle(); 
class1 = data(y == 1, :); 
class2 = data(y == -1, :); 
n = size(y); % number of measurements

% compute class means 
mu_1 = (1/size(class1,1))*ones(1,size(class1,2))*class1; 
mu_2 = (1/size(class2,1))*ones(1,size(class2,2))*class2; 

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
    sigma_2 = sigma_2 + (class2(i,:) - mu_1)'*(class2(i,:) - mu_1);
end

sigma_2 = (1/size(class2, 1))*sigma_2; 

% now classify!!
%set up priors 
p_1 = size(class1,1)/size(data,1);
p_2 = size(class2, 1)/size(data, 1); 

% compute sigma^(-1) * (x-mu): (NOTE THAT X AND MU ARE ROW VECTORS!!) 

z1 = sigma_1 \ (x - mu_1)'; 
z2 = sigma 2 \ (x - mu_2)'; 

% finally compute the decision rule!

p_x_1 = (1/sqrt(2*pi*det(sigma_1)^2))*exp(-0.5*(x-mu_1)*z)*p_1;
p_x_2 = (1/sqrt(2*pi*det(sigma_2)^2))*exp(-0.5*(x-mu_2)*z)*p_2; 

if p_x_1 > p_x_2 
    
    label = 1; 
else 
    label = -1;
end

end


    
    



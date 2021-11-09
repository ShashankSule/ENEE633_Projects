function label = BayesClassifier(x, class1, class2, mu_1, mu_2, sigma_1, sigma_2)
%% Let's do Bae's classification here. 

% Recall that we model each class as a Gaussian with ML parameters and 
% then do a bayes' classifier using i = argmax p(x | i)p(i) 

% Arguments: 
% x -- a training point

% Output: 
% label -- the bayes' classifier label 


% now classify!!

n = size(class1,1) + size(class2,2); % total # of training pts
%set up priors 
p_1 = size(class1,1)/n;
p_2 = size(class2,1)/n;  

% compute sigma^(-1) * (x-mu): (NOTE THAT X AND MU ARE ROW VECTORS!!) 

z1 = sigma_1 \ (x - mu_1)'; 
z2 = sigma_2 \ (x - mu_2)'; 

% finally compute the decision rule!

% here we'll have to circumvent the fact that we don't want to multiply by
% the reciprocal of the determinant of the covariance matrices

if cond(sigma_1) > cond(sigma_2) % compute |sigma_1 (sigma_2)^(-1)|
    prefactor = det(sigma_1 * (sigma_2 \ eye(size(sigma_2))));
    if exp(-0.5*(x-mu_1)*z1)*p_1 > prefactor*exp(-0.5*(x-mu_2)*z2)*p_2
        label = 1; 
    else
        label = -1;
    end
else                            % compute |sigma_2 (sigma_1)^(-1)|
    prefactor = det(sigma_2 * (sigma_1 \ eye(size(sigma_1))));
    if prefactor*exp(-0.5*(x-mu_1)*z1)*p_1 > exp(-0.5*(x-mu_2)*z2)*p_2
        label = 1; 
    else
        label = -1;
    end
end

end


    
    



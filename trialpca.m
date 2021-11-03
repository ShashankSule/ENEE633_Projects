%% generate circle data

t = linspace(0,1,100); 
x = [cos(2*pi*t); sin(2*pi*t); ones(4,100)];
% plot(x(1,:), x(2,:), 'bo');
x = x'; % put the observations in the rows 
%% do pca 
[coeff, score, latent] = pca(x); 

%% Lessons: 
% a. Columns of score are what you want
% b. X*coeff should be score 
% latent is good. 
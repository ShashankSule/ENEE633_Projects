%% Let's test the bayes' classifier here on DATA

% Compute labels here

[data, y] = loadandfiddle(); 
n = size(data,1);
[class1, class2, mu_1, mu_2, sigma_1, sigma_2] = preprocess(data,y); 
bayes_labels = zeros(size(data,1),1); 
for i=1:size(bayes_labels)
    bayes_labels(i) = BayesClassifier(data(i,:),class1, class2, ...
                                      mu_1, mu_2, sigma_1, sigma_2); 
end

%% Compute testing error 

indicator = abs(bayes_labels - y)/2; % form I_{y_n != F(x_n)}
trainingerr = (1/n)*sum(indicator); % compute training error


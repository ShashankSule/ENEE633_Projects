%% Let's test the bayes' classifier here on DATA

% Compute labels here

[data, y] = loadandfiddle(); 
bayes_labels = zeros(size(data,1),1); 
for i=1:size(bayes_labels)
    bayes_labels(i) = BayesClassifier(data(i,:)); 
end

% Compute testing error 




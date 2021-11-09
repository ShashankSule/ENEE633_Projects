%% let's do a supervised k-NN here! 

k = 10; 
[data, y] = loadandfiddle(); 
training = data; % initialize 
x = data(300,:); % testing point 
distances = zeros(size(training,1),1); %vector of distances
for i = 1:size(training,1)
    distances(i) = norm(x - training(i,:));
end

% sort the vector in ascending order 
[~,inds] = sort(distances, 'ascend'); 
k_labels = y(inds(1:10)); %find k nearest labels 
A = tabulate(k_labels); 
maximal_labels = find(A(:,3) == max(A(:,3))); % find the most frequent 
                                              % label among k-NN's
label = A(maximal_labels(1),1); %find the most popular label 

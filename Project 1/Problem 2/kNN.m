%% let's do a supervised k-nn here! 
[mda_data,~,y] = loadandfiddle(); 
%data = mda_data; 
props = linspace(0.1,0.9,9); 
n_neighbours = 2*(1:10); 
testing_err = zeros(size(n_neighbours,2), size(props,2));  
%% set up training and test data
for l=1:size(n_neighbours,2)
    for m=1:size(props,2)
        kk = n_neighbours(l);
        prop = props(m); 
        fprintf("Computing for %d neighbours with %d proportion of data \n", kk,prop); 
        %training = data; % initialize 
        %x = data(300,:); % testing point 
        n_classes = size(unique(y),1); 
        size_classes = sum(y==1);
        K = floor(prop*size_classes); % number of training points from each class 
        training_inds = repelem(size_classes*(0:(n_classes-1)) + 1, K) + repmat(0:(K-1), 1, 2);
        training = mda_data(training_inds, :); 
        training_labels = y(training_inds);
        test_inds = setdiff(1:size(mda_data,1),training_inds);
        test = mda_data(test_inds,:);
        test_labels = y(test_inds); 

        % the actual classifier
        %test = mda_data(302,:); 
        %k = K-1; 
        label = zeros(size(test,1),1); 
        for j=1:size(test,1)
            distances = zeros(size(training,1),1); %vector of distances

            for i = 1:size(training,1)
                distances(i) = norm(test(j,:) - training(i,:));
            end

            % sort the vector in ascending order 
            [~,inds] = sort(distances, 'ascend'); 
            k_labels = training_labels(inds(1:kk)); %find k nearest labels 
            A = tabulate(k_labels); 
            label(j) = A(find(A(:,3) == max(A(:,3)),1),1); % find the most frequent 
                                                          % label among k-NN's
        end
        % compute testing error
        testing_err(l,m) = mean(test_labels ~= label); 

        
    end
end


%% plot
set(0,'defaulttextinterpreter','latex')
imagesc(testing_err); 
xlabel("Proportion of training points per class", 'Interpreter', 'latex'); 
ylabel("Number of nearest neighbours ($k$)", 'Interpreter', 'latex');
colormap copper; 
xt = get(gca, 'XTick'); 
xtlbl = props; 
set(gca, 'XTick',xt, 'XTickLabel',xtlbl, 'XTickLabelRotation',30)
yt = get(gca, 'YTick'); 
ytlbl = n_neighbours; 
set(gca, 'YTick',yt, 'YTickLabel',ytlbl, 'XTickLabelRotation',0)  
c = colorbar;
c.Label.String = "Training error";



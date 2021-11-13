%% let's do kmeans here! 
load('hw06-data1.mat');
data = X'; 
Ks = [1; 2; 4; 8];
Labelz = zeros(4, size(data,1));
affinities = zeros(4,1);
for i = 1:4
    %% initialization 
    K = Ks(i); % # of clusters
    n = size(data,1); % # of samples
    centroids = data(randsample(n,K,false),:); % initial cluster centers
    K_old = zeros(K,n); % matrix of assignments
    for k = 1:n % set every column of K_old
        D = centroids - repmat(data(k,:),K,1); % D is K x d 
        distances = diag(D*D'); % vector of distances is tr(D D')  
        [~,I] = min(distances); % cluster assignment
        K_old(I,k) = 1; % set Ith element in the ith column of K_old to be 1; 
    end
    %% recomputing
    %recompute the cluster centers based on the initial assignment
    centroids = ((diag(K_old*ones(size(K_old,2),1)))^(-1))* K_old * data; 
    K_new = zeros(K,n); % matrix of assignments
    for k = 1:n % set every column of K_old
        D = centroids - repmat(data(k,:),K,1); % D is K x d 
        distances = diag(D*D'); % vector of distances is tr(D D')  
        [~,I] = min(distances); % cluster assignment
        K_new(I,k) = 1; % set Ith element in K_new to be 1; 
    end

    %% running K means 
    iter = 1; 
    while norm(K_new - K_old) > 1e-5
        iter = iter + 1; 
        K_old = K_new; % K_new is carried over from the previous step; now it becomes old :')
        centroids = ((diag(K_old*ones(size(K_old,2),1)))^(-1))* K_old * data; 
        for k = 1:n % set every column of K_old
            D = centroids - repmat(data(k,:),K,1); % D is K x d 
            distances = diag(D*D'); % vector of distances is tr(D D')  
            [~,I] = min(distances); % cluster assignment
            K_new(I,k) = 1; % set Ith element in K_new to be 1; 
        end
    end
    %% get labels for the data
    labels = zeros(size(data,1),1); 
    for j = 1:K 
        cluster = find(K_new(j,:) == 1); 
        labels(cluster) = j; 
    end
    Labelz(i,:) = labels; 
    
    %% get affinities 
    % compute cluster distances 
    total = 0.0; %initialize sum
    for j=1:Ks(i) %iterate thru clusters
      
        clustSum = 0.0; % initialize cluster sum 
        Nclust = sum(Labelz(i,:) == j); % size of jth cluster
        Cluster = data(Labelz(i,:) == j, :); % collect the jth cluster
        for k=1:size(Cluster,1) % iterate thru Cluster
            for m=1:size(Cluster,1) 
                clustSum = clustSum + norm(Cluster(m,:) - Cluster(k,:));
            end
        end
        total = total + (1/Nclust)*clustSum; 
    end
    affinities(i) = total; 
end

%% plot the data!!!

% figure();
% subplot(2,2,1); 
figure();
a1 = subplot(2,2,1); 
scatter(data(:,1), data(:,2), 30, Labelz(1,:), 'filled', ...
        'MarkerFaceAlpha', 0.5);
str = strcat("J = ", num2str(affinities(1))); 
annotation('textbox','String',str,'Position',a1.Position,'Vert','bottom','FitBoxToText','on');
colormap parula; 
set(gca, 'ycolor','w');
set(gca, 'xcolor','w');

a2 = subplot(2,2,2); 
scatter(data(:,1), data(:,2), 30, Labelz(2,:), 'filled', ...
        'MarkerFaceAlpha', 0.5);
str = strcat("J = ", num2str(affinities(2))); 
annotation('textbox','String',str,'Position',a2.Position,'Vert','bottom','FitBoxToText','on');

colormap parula; 
set(gca, 'ycolor','w');
set(gca, 'xcolor','w');

a3 = subplot(2,2,3); 
str = strcat("J = ", num2str(affinities(3))); 
scatter(data(:,1), data(:,2), 30, Labelz(3,:), 'filled', ...
        'MarkerFaceAlpha', 0.5);
annotation('textbox','String',str,'Position',a3.Position,'Vert','bottom','FitBoxToText','on');
colormap parula; 
set(gca, 'ycolor','w');
set(gca, 'xcolor','w');

a4 = subplot(2,2,4); 
str = strcat("J = ", num2str(affinities(4))); 
scatter(data(:,1), data(:,2), 30, Labelz(3,:), 'filled', ...
        'MarkerFaceAlpha', 0.5);
annotation('textbox','String',str,'Position',a4.Position,'Vert','bottom','FitBoxToText','on');
scatter(data(:,1), data(:,2), 30, Labelz(4,:), 'filled', ...
        'MarkerFaceAlpha', 0.5);
colormap parula; 
set(gca, 'ycolor','w');
set(gca, 'xcolor', 'w');


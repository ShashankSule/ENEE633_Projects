%% spectral clustering baby! 
load('hw06-data1.mat');
data = X';
n = size(data,1); % number of points
k = @(x,y) exp((-(norm(x-y))^2)/10); % kernel here
for i = 1:n
    for j = 1:n 
        K(i,j) = k(data(i,:), data(j,:)); % set up kernel matrix 
    end
end
K = K - eye(n,n); 
Ks = [1; 2; 4; 8];
Labelz = zeros(4, size(data,1)); 
affinities = zeros(4,1); 
%% compute here
for i=1:4 
    Labelz(i,:) = spectralcluster(K, Ks(i), 'Distance', 'precomputed');
    
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

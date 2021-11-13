%% let's load illumination data here 
illumination = load('illumination.mat');
raw_data = illumination.illum; 

%% Fiddle here
classes = {}; % store data of all classes 
recentered_classes = {}; % store recentered data
covariances = {}; % store covariances
means = {}; % store centers of all data
for i = 1:68
    classes{i} = raw_data(:,:,i)'; 
    mean = (1/size(classes{i},1))*ones(1,size(classes{i},1))*classes{i}; 
    recentered_classes{i} = classes{i} - ...
                            repmat(mean, size(classes{i},1),1); % recenter
    covariances{i} = recentered_classes{i}' * recentered_classes{i};
    % compute covariance
end

data = [];
recentered_data = []; 
y = []; 

for i = 1:68
    data = [data; classes{i}];
    recentered_data = [recentered_data; recentered_classes{i}];
    y = [y; i*ones(size(classes{i},1),1)];
end

%% prep for pca
centre = (1/size(data,1))*ones(1,size(data,1))*data; 
centered_data = data - repmat(centre, size(data,1), 1); 
covar = centered_data' * centered_data; 
 
%%  eigenvalue computation here
[V,D] = eigs(covar,50); 
proj_data = data*V; 

%% MDA here 
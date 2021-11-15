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
    avg = (1/size(classes{i},1))*ones(1,size(classes{i},1))*classes{i}; 
    recentered_classes{i} = classes{i} - ...
                            repmat(avg, size(classes{i},1),1); % recenter
    means{i} = avg; % compute class mean 
    covariances{i} = (1/size(classes{i},1))* ...
                     recentered_classes{i}' * recentered_classes{i};
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

avg = (1/size(data,1))*ones(1,size(data,1))*data; 

Sigma_b = zeros(size(covariances{1})); % between class scatter 
for i=1:68
    Sigma_b = Sigma_b + (1/size(classes{i},1))* ...
              (means{i} - avg)' * (means{i} - avg); 
end

Sigma_w = zeros(size(covariances{1})); % within class scatter
for i = 1:68
    Sigma_w = Sigma_w + (1/size(classes{i},1))*covariances{i};
end


%% prep for pca
centered_data = data - repmat(avg, size(data,1), 1); 
covar = centered_data' * centered_data; 
 
%%  eigenvalue computation here
[V,D] = eigs(covar,50); 
pca_data = data*V; 

%% MDA here
Sigma_w = Sigma_w + eye(size(Sigma_w)); 
[A, Sigmas] = eigs(Sigma_b, Sigma_w,20); 
mda_data = data*A;
%% plot 
%scatter3(mda_data(:,3), mda_data(:,4), mda_data(:,5), 30, y)

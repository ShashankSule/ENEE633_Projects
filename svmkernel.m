function label = svmkernel(training, y, test,k)
%% Here find the SVM for the projected data using the active set method
% The only difference between the kernel case and the linear case is that
% in the 
% linear case XX is the data matrix (because the feature map is always identity) 
% kernel case XX is the kernel matrix 
% Furthermore, the final map we get 
%% Setting up arguments for ASM.m 
% [training,y] = loadandfiddle(); %data matrix
%data = training(41:end, :); 
%y = y(41:end); 
data = training; 
y = double(y); 
y = 2*y - 3; % nominal to float conversion 
c = 100; %Constant in the soft margin penalty function 
n = length(y); % number of data points
%D = [(y*y').*((XX * XX')) zeros(n,n); zeros(n,n) zeros(n,n)]; %SPD matrix 
%                                                              in quadratic 
%           
% fprintf('number of training points: %d \n',size(training,1));
% fprintf('number of training labels: %d \n', size(y,1));
% fprintf('Number of 1s: %d \n', sum(y==1));
% fprintf('Number of -1s: %d \n', sum(y==-1));
% set kernel 
if ~exist('k', 'var')
    sigma = 100; % variance parameter 
    k = @(x,y) exp(-(norm(x-y)^2)/(2*sigma));
end

XX = zeros(n,n); 
% form kernel matrix 
for i = 1:n
    for j = 1:n
        XX(i,j) = k(data(i,:), data(j,:)); % calculate k(x_i, x_j) 
    end
end

% compute [D}_ij = y_i y_j k(x_i, x_j); after this rest of the steps 
% are the same as the linear case 

D = (y*y').*XX;


%reducing to n-1 vars: we must do this since ASM.m only takes in active
%constraints

yn = y(end);
D00 = D(1:end-1,1:end-1);
D01 = D(1:end-1,end);
D10 = D(end,1:end-1);
D11 = D(end,end);
%new matrix
Htil = D00 + y(1:n-1)*D11*y(1:n-1)' - (1/yn)*y(1:n-1)*D10 ...
       - (1/yn)*D01*y(1:n-1)'; % new hessian 
d = ones(n-1, 1) - (1/yn)*y(1:n-1); % new first order term 
cons = (-1/yn)*y(1:n-1)'; % last row of lhs of constraints
                          % lambda_n row vector 
C = [eye(n-1,n-1); cons]; % lhs of constraints
b = zeros(n,1); % rhs of constraints

%set of active constraints at the initial point
x = zeros(n-1,1); % the initialization is zeros everywhere
W = 1:n; % at the initial point all constraints are active
W = W'; % making the set of active constraints a column vector 
Htil = Htil - (min(eig(Htil)) - 10)*eye(n-1,n-1); % Make sure to avoid shitty conditioning
gfun = @(x)Htil*x - d; %gradient function
Hfun = @(x)Htil; %hessian function 
%% Time to run the solver! 
fprintf('Norm of initial hessian: %d \n', norm(Htil(:)));
fprintf('Norm of initial gradient %d \n', norm(d));
[lambs, ~] = ASM(x, gfun, Hfun, C, b, W,100); % run asm 

%% Clean up output
soln = lambs(:,end); % extracting the lambda vector
lambend = cons*soln; % extract the last lambda
soln = [soln; lambend]; % put all the lambda's together
%fprintf('Computed full solution!\n'); 
%opt = zeros(n,1); 
%pos_lams = find(soln > 1e-6); %find all the lambdas that are actually pos
%opt(pos_lams) = soln(pos_lams); %opt is now the vector of all the positive lambdas
wASM = (XX')*(y .* soln); % wASM here is phi, the vector of evaluations in
                          % the hilbert space
%fprintf('Computed phi!\n');
%% Computing B via support vectors
%
% let x+ and x- be the points in the + and - classes with the largest
% lambdas. then b = 1 - phi(x+) and b = -1 + phi(x-) so b = - (phi(x+) +
% phi(x-) / 2)  
plus_vec = find(soln == max(soln(y==1)),1,'first');
minus_vec = find(soln == max(soln(y==-1)), 1, 'first'); 
B = -0.5*(wASM(plus_vec) + wASM(minus_vec));
fprintf('Computed constant!\n'); 
%% Compute training error!
wASM = wASM + B;
proj_vals = sign(wASM); % form phi(x_n) + b > 0 vector
indicator = abs(proj_vals - y)/2; % form I_{y_n != F(x_n)}
trainingerr = (1/n)*sum(indicator);
fprintf('Computed training error!\n');
%% plot the surface! 

% % figure out x,y, and z limits of the plot 
% xlim = min(XX(:,1)); 
% xLim = max(XX(:,1)); 
% ylim = min(XX(:,2)); 
% yLim = max(XX(:,2));  
% X = linspace(xlim, xLim, 2); 
% Y = linspace(ylim, yLim, 2); 
% [xx, yy] = meshgrid(X,Y); 
% % equation for the plane is z = (-theta1/theta3)x + (-theta2/theta3) + b
% 
% zz = (-wASM(1)/wASM(end-1))*xx + (-wASM(2)/wASM(end-1))*yy + wASM(end)*ones(size(xx)); 
% %% Plot! 
% 
% surf(xx,yy,zz, 'FaceAlpha',0.5, 'FaceColor', 'b', 'EdgeColor', 'none')
% hold on; 
% scatter3(XX(:,1), XX(:,2), XX(:,3), 20, 10*y);
% colormap(flag);
% colorbar 
%% compute label of test data
%test = training(200:240,:); 
K = zeros(n,size(test,1));
fprintf('Now testing...\n'); 
for j = 1:size(test,1)
    for i=1:n
        K(i,j) = k(test(j,:),training(i,:)); % K is [K(x,x_i)] vector
    end
end
fprintf('Now computing labels\n'); 
label = double(sign(B + K'*(y.*soln))); 
end
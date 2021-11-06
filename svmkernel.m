%% Here find the SVM for the projected data using the active set method
% The only difference between the kernel case and the linear case is that
% in the 
% linear case XX is the data matrix (because the feature map is always identity) 
% kernel case XX is the kernel matrix 
% Furthermore, the final map we get 
%% Setting up arguments for ASM.m 
[data,y] = loadandfiddle(); %data matrix
c = 100; %Constant in the soft margin penalty function 
n = length(y); % number of data points
%D = [(y*y').*((XX * XX')) zeros(n,n); zeros(n,n) zeros(n,n)]; %SPD matrix 
%                                                              in quadratic 
%                                                              program 

% set kernel 
sigma = 0.1; % variance parameter 

k = @(x,y) exp(-(norm(x-y)^2)/(2*sigma));
XX = zeros(n,n); 
% form kernel matrix 
for i = 1:n
    for j = 1:n
        XX(i,j) = k(data(i,:), data(j,:)); % calculate k(x_i, x_j) 
    end
end

% compute [D}_ij = y_i y_j k(x_i, x_j); after this rest of the steps 
% are the same as 
D = (y*y').*((XX * XX'));


% d = ones(n,1);
% Ap = [eye(n,n) eye(n,n)]; % the matrix A'
% App = y';
% C = [eye(n,n); -eye(n,n); App];  %matrix of contraints
% b = [zeros(2*n + 1,1); c*ones(n,1)]; % vector of constraints
% b = [zeros(n,1); -c*ones(n,1); 0];
% W = [1:n 2*n + 1];
% W = W';
% init = zeros(n,1);



%reducing to n-1 vars: we must do this since ASM.m only takes in active
%constraints

yn = y(end);
D00 = D(1:end-1,1:end-1);
D01 = D(1:end-1,end);
D10 = D(end,1:end-1);
D11 = D(end,end);
%new matrix
Htil = D00 + y(1:n-1)'*D11*y(1:n-1) - (1/yn)*y(1:n-1)*D10 ...
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
Htil = Htil + 10*eye(n-1,n-1); % Make sure to avoid shitty preconditioning
gfun = @(x)Htil*x - d; %gradient function
Hfun = @(x)Htil; %hessian function 
%% Time to run the solver! 
[lambs, lm] = ASM(x, gfun, Hfun, C, b, W); % run asm 

%% Clean up output
soln = lambs(:,end); % extracting the lambda vector
lambend = cons*soln; % extract the last lambda
soln = [soln; lambend]; % put all the lambda's together
opt = zeros(n,1); 
pos_lams = find(soln > 1e-6); %find all the lambdas that are actually pos
opt(pos_lams) = soln(pos_lams); %opt is now the vector of all the positive lambdas
 
%computing b: 
wASM = (XX')*(y .* soln); % wASM here is phi, the vector of evaluations
slack_var = find(opt > 0, 1); % pick a support vector
B = (1/y(slack_var)*soln(slack_var)) - wASM(slack_var); % use comp slack

% Computing B via soft margin support vectors
% avg = XX(soln == max(soln(y==1)),:) ...
  + XX(soln == max(soln(y==-1)),:);
% avg = XX(abs(soln(y==1)-(c/2)) == min(abs(soln(y==1)-(c/2))),:) ...
%       + XX(abs(soln(y==-1)-(c/2)) == min(abs(soln(y==-1)-(c/2))),:); 
% B = -0.5*(avg * wASM);
wASM = wASM + B; 
% wASM = [wASM; B];
% wASM = real(wASM); 
%% Compute Training error 

proj_vals = 2*(wASM > 0) - 1; % form phi(x_n) + b > 0 vector
indicator = abs(proj_vals - y)/2; % form I_{y_n != F(x_n)}
trainingerr = (1/n)*sum(indicator); % compute training error

%% plot the plane! 

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
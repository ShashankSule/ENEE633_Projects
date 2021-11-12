function [theta, B, trainingerr] = WeakClassifier(p, training, y,k, tol)
%% let's write a weak classifier selector here 
%Arguments: 
% p - probability weight vector 
% training - training data
% y - training labels 
% k -kernel function 
% tol - the error parameter; the classifier is chosen on the basis of 
%       training error < 1/2 - tol 
% [training, y] = loadandfiddle(); 
% p = (1/size(training,1))*ones(size(training,1),1); 

if ~exist('tol','var')
    tol = 1e-2;
end

%% Set up arguments for ASM 
n = length(y); % number of data points
%D = [(y*y').*((XX * XX')) zeros(n,n); zeros(n,n) zeros(n,n)]; %SPD matrix 
%                                                              in quadratic 
%                                                              program 

if ~exist('k', 'var')
    % set kernel 
    sigma = 0.1; % variance parameter 

    k = @(x,y) exp(-(norm(x-y)^2)/(2*sigma));
    XX = zeros(n,n); 
end

% form kernel matrix 
for i = 1:n
    for j = 1:n
        XX(i,j) = k(training(i,:), training(j,:)); % calculate k(x_i, x_j) 
    end
end

% compute [D}_ij = y_i y_j k(x_i, x_j); after this rest of the steps 
% are the same as 
D = (y*y').*((XX * XX'));

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
Htil = Htil + 10*eye(n-1,n-1); % Make sure to avoid shitty preconditioning
gfun = @(x)Htil*x - d; %gradient function
Hfun = @(x)Htil; %hessian function 

%% Figuring out the weak classifier 
% Now the way we figure out the good solution is as follows: 
% until sum p_n I_{yn != phi(x_n, theta)} becomes less than 1/2 we keep
% updating

trainingerr = 1;
while trainingerr > (0.5 - tol)
    [lambs, ~] = ASM(x, gfun, Hfun, C, b, W, 100); % run asm
    
    % clean up output to compute the classifier
    soln = lambs(:,end); % extracting the lambda vector
    lambend = cons*soln; % extract the last lambda
    soln = [soln; lambend]; % put all the lambda's together
    opt = zeros(n,1); 
    pos_lams = find(soln > 1e-10); %find all the lambdas that are actually pos
    opt(pos_lams) = soln(pos_lams); %opt is now the vector of all the positive lambdas
    
    %computing b: 
    wASM = (XX')*(y .* soln); % wASM here is phi, the vector of evaluations
    slack_var = find(opt > 0, 1); % pick a support vector
    B = (1/y(slack_var)*soln(slack_var)) - wASM(slack_var); % use comp slack
    wASM = wASM + B; 
    % compute training error 
    proj_vals = 2*(wASM > 0) - 1; % form phi(x_n) + b > 0 vector
    indicator = abs(proj_vals - y)/2; % form I_{y_n != F(x_n)}
    trainingerr = sum(p.*indicator); % compute training error
    
    % prepare for next iteration 
    x = soln(1:n-1); % pass current solution as initialization to the solver
    W = find(x == 0.0); % find the current set of active constraints, i.e
                      % lambda = 0
    fprintf('Training error is %d \n', trainingerr); 
end

%% now we'll output the classifier as sum y_i lambda_i K(. , x_i) + theta0

theta = y .* soln; % this is the coordinate vector in the {K(. , x_i)}_i
                   % basis
end





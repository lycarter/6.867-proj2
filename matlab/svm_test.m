function svm_test(name)
disp('======Training======');
% load data from csv files
data = importdata(strcat('data/data_',name,'_train.csv'));
X = data(:,1:2);
Y = data(:,3);

    function prod = K(x1,x2)
        sigma = 1;
        prod = exp(-sum((x1-x2).^2)/(2*sigma^2));
    end

% Carry out training, primal and/or dual
[theta, theta0, alphas] = svmKernal(X, Y, .01, @K);

% Define the predictSVM(x) function, which uses trained parameters
function out = predictSVM(x)
    
    %out = (theta*x + theta0);
    out = theta0;
    M = find(alphas > .00001);
    for i = M'
        out = out + alphas(i).*Y(i).*K(X(i,:),x');
    end
end

1

hold on;
% plot training results
%plotDecisionBoundary(X, Y, @predictSVM, [-1, 0, 1], 'SVM Train');


disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

% plot validation results
%plotDecisionBoundary(X, Y, @predictSVM, [-1, 0, 1], 'SVM Validate');
end

function [theta, theta0, alphas] = svmKernal(X, Y, C, K)

optim_ver = ver('optim');
optim_ver = str2double(optim_ver.Version);
if optim_ver >= 6
    opts = optimset('Algorithm', 'interior-point-convex');
else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
end

n = size(X,1);
d = size(X,2);

H = zeros(n);

for i = 1:n
    for j = 1:n
        H(i,j) = Y(i)*Y(j)*K(X(i,:),X(j,:));
    end
end

f = -1*ones(1,n);

Aeq = Y';

beq = 0;

lb = zeros(n,1);
ub = C*ones(n,1);
A = [];
b = [];
x0 = [];

alphas = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, opts);

theta = (alphas.*Y)'*X;

M = find(alphas > .00001);
size(M)
theta0 = 0;
for j = M'
    theta0 = theta0 + Y(j);
    for i = 1:n
        theta0 = theta0 - K(X(j,:), X(i,:))*(alphas(i)*Y(i));
    end
end
theta0 = (1/size(M,1))*theta0;
end

function [theta, theta0] = svmGood(X, Y, C)

optim_ver = ver('optim');
optim_ver = str2double(optim_ver.Version);
if optim_ver >= 6
    opts = optimset('Algorithm', 'interior-point-convex');
else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
end

n = size(X,1);
d = size(X,2);

H = zeros(n);

for i = 1:n
    for j = 1:n
        H(i,j) = Y(i)*Y(j)*dot(X(i,:),X(j,:));
    end
end

f = -1*ones(1,n);

Aeq = Y';

beq = 0;

lb = zeros(n,1);
ub = C*ones(n,1);
A = [];
b = [];
x0 = [];

alphas = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, opts);

theta = (alphas.*Y)'*X;

M = find(alphas > .00001);
size(M)
theta0 = 0;
for j = M'
    theta0 = theta0 + Y(j) - sum(X(j,:)*X'*(alphas.*Y))
end
theta0 = (1/size(M,1))*theta0;
end
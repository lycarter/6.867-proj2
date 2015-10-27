function svm_test(name)
disp('======Training======');
% load data from csv files
data = importdata(strcat('data/data_',name,'_train.csv'));
X = data(:,1:2);
Y = data(:,3);

% Carry out training, primal and/or dual
[theta, theta0] = svmGood(X, Y, 1);

% Define the predictSVM(x) function, which uses trained parameters
    function predictSVM(x)
        return sign(theta'*x + theta0)
        


hold on;
% plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], 'SVM Train');


disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

% plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], 'SVM Validate');
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
exI = find(alphas == max(alphas));
theta0 = Y(exI) - X(exI,:)*theta';
end
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
function [theta, theta0] = svmBad(X, Y, C)

optim_ver = ver('optim');
optim_ver = str2double(optim_ver.Version);
if optim_ver >= 6
    opts = optimset('Algorithm', 'interior-point-convex');
else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
end

X_positive = X(Y == 1, :);
X_negative = X(Y == -1, :);

n = size(X,1);
d = size(X,2); 

% plot the positive and negative points using the first two features as
% (x1, x2) => (x,y)
% hold on
% plot(X_positive(:,1),X_positive(:,2),'or')
% plot(X_negative(:,1),X_negative(:,2),'+b')

H = eye(n);

f = -1*ones(d,1);

A = [];
b = [];

Aeq = ones(1,n);
beq = 0;

lb = 0;
ub = C;

x0 = [];

alphas = quadprog(H,f,A,b, Aeq, beq, lb, ub, x0, opts);

constraints = find(alphas ~= 0);
M = size(constraints,2);

theta = alphas.*Y.*X;
theta0 = (1/M)*sum(Y(constraints) - X(constraints).'*sum(alphas.*Y.*X));
end

function xyz = toMin(X, Y, alphas)
term1 = sum(alphas);
term2 = .5*sum((alphas.'*alphas).*(Y.'*Y).*(X.'*X));
xyz = -(term1 - term2);
end
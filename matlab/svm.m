function [theta, theta0] = svm(X,Y)

optim_ver = ver('optim');
optim_ver = str2double(optim_ver.Version);
if optim_ver >= 6
    opts = optimset('Algorithm', 'interior-point-convex');
else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
end

n = size(X,1);
d = size(X,2);

X_Positive = X(Y==1,:);
X_Negative = X(Y==-1,:);

% plot the positive and negative points using the first two features as
% (x1, x2) => (x,y)
hold on
plot(X_Positive(:,1),X_Positive(:,2),'or')
plot(X_Negative(:,1),X_Negative(:,2),'+b')

%%% Code to find the SVM hyperplane
H=eye(d+1);
H(d+1,d+1)=0;
f=zeros(d+1,1);
Z = [X ones(n,1)];
A=-diag(Y)*Z;
c=-1*ones(n,1);

thetas=quadprog(H,f,A,c, [], [], [], [], [], opts);

%%% Code to plot the SVM separating hyperplane
X1=[-2:.1:2];
theta=thetas(1:d,1);
theta0=thetas(d+1,1);

w1 = thetas(1,1);
w2 = thetas(2,1);
b = theta0;
Y1=-(w1*X1+b)/w2; %Seperating hyperplane
plot(X1,Y1,'k-')
%%% Code to plot the SVM margins goes here! %%%
YUP=(1-w1*X1-b)/w2; %Margin
plot(X1,YUP,'m:')
YLOW=(-1-w1*X1-b)/w2; %Margin
plot(X1,YLOW,'m:') 
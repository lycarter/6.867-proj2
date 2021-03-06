function lr_test(name)
disp('======Training======');
% load data from csv files

data = importdata(strcat('../hw2_resources/data/data_',name,'_train.csv'));

X = data(:,4:5);
Y = data(:,12);

% Carry out training.
%%% TODO %%%
mn = zeros(1,2);
st = zeros(1,2);
for i = 1:2
    mn(1,i) = mean(X(:,i));
    st(1,i) = std(X(:,i));
    X(:,i) = (X(:,i) - mn(1,i)) / st(1,i);
end
    
mint = gradDescentLR([2, 0,-5], 0.01, 0.01, X, Y, 1);
% Define the predictLR(x) function, which uses trained parameters
%%% TODO %%%
%mint = [10000, 5, 1000];
%mint = [2, 0, -0.8];
mint
function label = predictLR(x)
    l = 1.0 / (1 + exp(-(x' * mint(1, 1:2)' + mint(1, 3))));
    if l > 0.5
        label = 1;
    else
        label = 0;
    end
end


hold on;

% plot training results
plotDecisionBoundary(X, Y, @predictLR, [0, 0.5, 1], 'LR Train');

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('../hw2_resources/data/data_',name,'_validate.csv'));
X = validate(:,4:5);
Y = validate(:,12);
for i = 1:2
    X(:,i) = (X(:,i) - mn(1,i)) / st(1,i);
end
% plot validation results
plotDecisionBoundary(X, Y, @predictLR, [0.5, 0.5, 0.5], 'LR Validate');
end

function lr_test(name)
disp('======Training======');
% load data from csv files

data = importdata(strcat('../hw2_resources/data/data_',name,'_train.csv'));

X = data(:,1:2);
Y = data(:,3);

% Carry out training.
%%% TODO %%%
mint = gradDescentLR([1, 1, 1], 0.001, 0.008, X, Y, 30);
% Define the predictLR(x) function, which uses trained parameters
%%% TODO %%%
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
plotDecisionBoundary(X, Y, @predictLR, [0.5, 0.5], 'LR Train');

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('../hw2_resources/data/data_',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

% plot validation results
plotDecisionBoundary(X, Y, @predictLR, [0.5, 0.5], 'LR Validate');
end

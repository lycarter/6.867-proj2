function plotDecisionBoundary(X, Y, scoreFn, values, mytitle, firstcol, secondcol)
% X is data matrix (each row is a data point)
% Y is desired output (1 or -1)
% scoreFn is a function of a data point
% values is a list of values to plot

% Plot the decision boundary. For that, we will asign a score to
% each point in the mesh [x_min, m_max]x[y_min, y_max].
    
mins=min(X)-1;
maxes=max(X)+1;

avgs = mean(X);
    
% h = max((maxes(1)-mins(1))/200., (maxes(2)-mins(2))/200.);
h1 = (maxes(firstcol)-mins(firstcol))/50.;
h2 = (maxes(secondcol)-mins(secondcol))/50.;
% h = max((maxes(firstcol)-mins(firstcol))/50., (maxes(secondcol)-mins(secondcol))/50.);

[xx, yy] = meshgrid(mins(firstcol):h1:maxes(firstcol), mins(secondcol):h2:maxes(secondcol));

size(xx)

arr = ones(numel(xx),1)*avgs;

arr(:,firstcol) = reshape(xx,numel(xx), 1);
arr(:,secondcol) = reshape(yy, numel(yy), 1);

%arr=[xx(:),yy(:)];
zz = zeros(length(arr),1);
<<<<<<< HEAD
for i=1:length(arr)
    zz(i) = scoreFn(arr(i,:)'); 
=======
for i=1:length(arr),
    zz(i) = scoreFn(arr(i,:)');
>>>>>>> aa77a4445ee3f427dd0c86dadf0c668c3b967914
end  
zz=reshape(zz,size(xx));
   
figure;
hold on;
title(mytitle);
colormap cool
[C,h]=contour(xx, yy, zz, values);
set(h,'ShowText','on');
%Plot the training points
<<<<<<< HEAD
scatter(X(:,firstcol),X(:,secondcol),50,1-Y);
=======
X = X + rand(size(X))*.3-.15;
scatter(X(:,1),X(:,2),50,1-Y);
%axis([-1, 2, 0, 80]);
end
>>>>>>> aa77a4445ee3f427dd0c86dadf0c668c3b967914

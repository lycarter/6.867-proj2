function min = gradDescentLR(guess, step, threshold, X, Y)
% call with gradDescent(@quad, [-108, 646], .008, .01)
xn = guess;
x0 = zeros(size(guess));
iterations = 0;
while abs(f(x0, X, Y) - f(xn, X, Y)) > threshold
    iterations = iterations + 1;
    grad = finGrad(xn, step/10, X, Y);
    x0 = xn;
    xn = xn - step*grad;
end
iterations
min = xn;
end

function grad = finGrad(x, delta, X, Y)
d = delta/2;
dims = size(x);
vdim = dims(2);
empty = zeros(dims);
z = zeros(dims);
for i = 1:vdim
    temp = empty;
    temp(i) = 1;
    z(1,i) = f(x+d*temp, X, Y) - f(x-d*temp, X, Y);
end
grad = z./delta;
end

function sum = f(w, X, Y)
sum = 0;
for i = 1:size(X, 1)
    size(X, 2);
    a = X(i, 1:size(X, 2)) * w(1, 1:size(X, 2))';
    up = -Y(i, 1) * (a + w(1, end));
    sum = sum + log(1 + exp(up));
end
sum = sum + 10 * w * w';
end
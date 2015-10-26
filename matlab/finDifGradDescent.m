function min = finDifGradDescent(f, guess, step, threshold)
% call with finDifGradDescent(@quad, [-108, 646], .008, .01)
xn = guess;
x0 = zeros(size(guess));
iterations = 0;
while abs(f(x0) - f(xn)) > threshold
    iterations = iterations + 1;
    grad = finGrad(f, xn, step/10);
    x0 = xn;
    xn = xn - step*grad;
end
iterations
min = xn;
end

function grad = finGrad(f, x, delta)
d = delta/2;
dims = size(x);
vdim = dims(2);
empty = zeros(dims);
z = zeros(dims);
for i = 1:vdim
    temp = empty;
    temp(i) = 1;
    z(i) = f(x+d*temp) - f(x-d*temp);
end
grad = z./delta;
end

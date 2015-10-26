function min = gradDescent(f, g, guess, step, threshold)
% call with gradDescent(@f, @g, [-108, 646], .008, .0000001)
xn = guess;
x0 = zeros(size(guess));
iterations = 0;
while abs(f(x0) - f(xn)) > threshold
    iterations = iterations + 1;
    grad = g(xn);
    x0 = xn;
    xn = xn - step*grad;
end
min = xn;
end
% to plot the solutions as they are computed
opts = odeset('Stats','on','OutputFcn',@odephas3);
[t,y] = ode15s(@fast_slow1, [0 650], [1.4; 1.25; -0.58], opts);
xlabel('x')
ylabel('y')
zlabel('u')

% values for lambda: 0.878496, 0.988496, 0.992319, 0.998496
function dydt = fast_slow1(~,y)
dydt = [-y(1)+2*y(2)-y(2)^3/3+y(3);
    -y(2)+2*y(1)-y(1)^3/3+y(3);
    0.05*(0.878496-(y(1)+y(2))/2)];
end
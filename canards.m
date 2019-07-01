x = -3:0.01:3;
y1 = x;
z1 = (x.^3-3*x)/3;
plot(x,z1)
grid on
hold on

y2 = (-sqrt(3*(12-x.^2))-x)/2;
z2 = (2*x.^3-3*sqrt(3*(12-x.^2))-15*x)/6;

y2 = (sqrt(3*(12-x.^2))-x)/2;
z2 = (2*x.^3+3*sqrt(3*(12-x.^2))-15*x)/6;

[t,y] = ode45(@fast_slow1, [0 800], [2.3; 2.25; 1]);

plot(y(:,1),y(:,3))


% values for lambda: 0.994318, 0.993496, 0.993296
function dydt = fast_slow1(~,y)
dydt = [-y(1)+2*y(2)-y(2)^3/3+y(3);
    -y(2)+2*y(1)-y(1)^3/3+y(3);
    0.05*(0.993496-(y(1)+y(2))/2)];
end
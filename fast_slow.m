x = -3:0.01:3;
y1 = x;
z1 = (x.^3-3*x)/3;
plot3(x,y1,z1)
grid on
hold on

y2 = (-sqrt(3*(12-x.^2))-x)/2;
z2 = (2*x.^3-3*sqrt(3*(12-x.^2))-15*x)/6;
plot3(x,y2,z2)
hold on

y2 = (sqrt(3*(12-x.^2))-x)/2;
z2 = (2*x.^3+3*sqrt(3*(12-x.^2))-15*x)/6;
plot3(x,y2,z2)
hold on

[t,y] = ode45(@fast_slow1, [0 70000], [-2; 3.5; 2]);
plot3(y(:,1),y(:,2),y(:,3),'->')

function dydt = fast_slow1(~,y)
dydt = [-y(1)+2*y(2)-y(2)^3/3+y(3);
    -y(2)+2*y(1)-y(1)^3/3+y(3);
    -0.0001*(y(1)+y(2))/2];
end
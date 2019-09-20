x = -3:0.01:3;
y = 2*x-x.^3/3;
plot3(x,y,y)
grid on
hold on

[t,y] = ode45(@fast_slow1, [0 10], [1.42; 1.89; 1.89]);
plot3(y(:,1),y(:,2),y(:,3),'->')

function dydt = fast_slow1(~,y)
dydt = [0;
    -y(2)+2*y(1)-y(1)^3/3+y(3);
    -y(3)+2*y(1)-y(1)^3/3+y(3)];
end
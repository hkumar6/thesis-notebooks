% Set x-coordinate range
x = -3:0.1:3;

% Plot C_{0,1}, consensus critical manifold
y1 = x;
z1 = (x.^3-3*x)/3;
plot3(x,y1,z1,'LineWidth',2)
txt_c01 = '\leftarrow C_{0,1}';
txt_x = 2.5; txt_z = (txt_x^3-3*txt_x)/3;
text(txt_x, txt_x, txt_z, txt_c01, 'fontsize',18);
xlabel('x'); ylabel('y'); zlabel('u')
grid on; hold on

% Plot C_{0,2}
y2 = (-sqrt(3*(12-x.^2))-x)/2;
z2 = (2*x.^3-3*sqrt(3*(12-x.^2))-15*x)/6;
plot3(x,y2,z2,'LineWidth',2)
txt_c02 = '\leftarrow C_{0,2}';
txt_x = 2.5;
txt_y = (-sqrt(3*(12-txt_x^2))-txt_x)/2;
txt_z = (2*txt_x^3-3*sqrt(3*(12-txt_x^2))-15*txt_x)/6;
text(txt_x, txt_y, txt_z, txt_c02, 'fontsize',18);
hold on

% Plot C_{0,3}
y2 = (sqrt(3*(12-x.^2))-x)/2;
z2 = (2*x.^3+3*sqrt(3*(12-x.^2))-15*x)/6;
plot3(x,y2,z2,'LineWidth',2)
txt_c03 = '\leftarrow C_{0,3}';
txt_x = -2.5;
txt_y = (sqrt(3*(12-txt_x^2))-txt_x)/2;
txt_z = (2*txt_x.^3+3*sqrt(3*(12-txt_x^2))-15*txt_x)/6;
text(txt_x, txt_y, txt_z, txt_c03, 'fontsize',18);
hold on

% Initialize from a point
% Uncomment below to choose a point very close to C_{0,1}
% p1 = [2.4, 2.4, 0.4];
% An initial point close to the repelling part of C_{0,3}
p2 = [-2, 3.5, 1];
[t,y] = ode45(@vdp_3d, [0 500], p2);
plot3(y(:,1),y(:,2),y(:,3),'-','LineWidth',5,'Color','magenta')
hold on

% An initial point close to the repelling part of C_{0,1}
p3 = -[2.2, 2.4, 1.8];
[t1,y1] = ode45(@vdp_3d, [0 500], p3);
plot3(y1(:,1),y1(:,2),y1(:,3),'-','LineWidth',5,'Color','green')

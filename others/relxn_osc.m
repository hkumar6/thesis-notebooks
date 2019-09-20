function dydt = relxn_osc(~,y)
A = [[0 1 1 1];
 [1 0 0 0];
 [0 1 0 0];
 [1 1 0 0]];
D = diag([3,1,1,2]);
s = @(x,u) 2.*x - x.^3/3 + u;
dydt = [-D*y(1:end-1) + A*s(y(1:end-1),y(end)); -0.05*mean(y(1:end-1))];
end
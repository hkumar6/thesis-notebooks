% van der Pol oscillator on a 2-node digraph
function dydt = vdp_3d(~,y)
dydt = [-y(1)+2*y(2)-y(2)^3/3+y(3);
    -y(2)+2*y(1)-y(1)^3/3+y(3);
    -0.05*(y(1)+y(2))/2];
end
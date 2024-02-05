close all; clear; clc;
figure(1);
l=5;
syms acty;
proa = -0.4; prob = 1 ; proc = 5.3;
actx = linspace(0,6,30); 
%actx = actx(5+1:end-5);
actz = linspace(1.28,1.28,30); 
%actz = actz(5+1:end-5);
acty = proa.*(actx.^2) + prob.*actx + proc; 
%acty = acty(l+1:end);

plot3(actx,actz,acty,'<k')
a1 = dlmread("xp2.txt"); %a1(:,1) = [];
b1 = dlmread("yp2.txt"); %b1(:,1) = [];
hold on
%plot(a1(5,[1:25]),b1(5,[1:25]));
for i = 1:length(a1)
   plot3(a1(i,:),actz,b1(i,:),':b')
   hold on
end

xlabel("X(meters)"); ylabel("Y(meters)"); zlabel("Z(meters)");
grid on
axis([0 6 0 6 0 max(acty)]);
legend({'Actual Trajectory','Predicted Trajectories'},'Location','northeast');
close all; clear; clc;

syms y;
a = -1; b = -0.1; c = 3.23;
x = linspace(0,20,100);
z = linspace(3,3,100);
y = a.*(x.^2) + b.*x + c;

plot(x,y);
axis([0 6 0 max(y)])
grid on
xlabel("X"); ylabel("Z");
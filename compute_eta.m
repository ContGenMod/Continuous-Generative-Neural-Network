
clear; close all; clc;

% Wavelet di Daubechies with N vanishing moments 
N = 10;
wname = strcat('db',num2str(N));

% phi scaling function and belongs to V_0 (j = 0)
% iter is proportional to the precision of eta.
iter = 8;
[phi,psi,xval] = wavefun(wname,iter);
L = length(xval);
step = xval(2)-xval(1);
step_quadro = step*step;

% Adding padding to phi
phi_padding = zeros(3*L);
phi_padding(L+1:2*L) = phi(1:L);

path = '/data/silvias/matlab/eta_values/csv/';


eta = {};
for l=floor(-L*step):floor((3*L*step)*2)
    eta_aux = 0;
    for i=1:3*L
        for j=1:3*L
            k = floor(j-2*i+l/step);
            if k > 0 && k <= 3*L
                eta_aux = eta_aux + phi_padding(i)*phi_padding(j)*phi_padding(k)*step_quadro;
            end
        end
    end
    eta = [eta,eta_aux];
end
eta = cell2mat(eta);
%{
x=1:length(eta);
figure
plot(x,eta)
%}
if strcmp(wname,'db1')
    eta = eta(3:9);
elseif strcmp(wname,'db2')
    eta = eta(8:17);
elseif strcmp(wname,'db6')
    eta = eta(31:47);
elseif strcmp(wname,'db10')
    eta = eta(53:77);
end
%{
x=1:length(eta);
figure
plot(x,eta)
%}
writematrix(eta,strcat(path,'eta_db',num2str(N),'.csv'))

% ------------------------------------------------------------------------------------------------------------------------------------

eta_bar = {};
for l=floor(-2*L*step):floor((3*L*step)*2)
    eta_aux = 0;
    for i=1:3*L
        for j=1:3*L
            k = floor(2*j-2*j+l/step);
            if k > 0 && k <= 3*L
                eta_aux = eta_aux + phi_padding(i)*phi_padding(j)*phi_padding(k)*step_quadro;
            end
        end
    end
    eta_bar = [eta_bar,eta_aux];
end
eta_bar = cell2mat(eta_bar);
%{
x=1:length(eta_bar);
figure
plot(x,eta_bar)
%}
if strcmp(wname,'db1')
    eta_bar = eta_bar(4:8);
elseif strcmp(wname,'db2')
    eta_bar = eta_bar(10:14);
elseif strcmp(wname,'db6')
    eta_bar = eta_bar(33:42);
elseif strcmp(wname,'db10')
    eta_bar = eta_bar(57:68);
end
%{
x=1:length(eta_bar);
figure
plot(x,eta_bar)
%}
writematrix(eta_bar,strcat(path,'eta_bar_db',num2str(N),'.csv'))

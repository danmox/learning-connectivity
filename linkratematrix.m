function R = linkratematrix(x)
% LINKRATEMATRIX return the predicted channel mean and variance between
% each agent in the team configuration x
%
% inputs:
%   x - 2Nx1 column vector of N [x;y] agent positions stacked
%
% outputs:
%   R.avg - NxN matrix with R.avg(i,j) representing the expected channel
%           rate between agents i,j with R.avg == 0 when i == j
%   R.var - NxN matrix with R.var(i,j) representing the variance in the
%           expected channel rate between agents i,j with R.avg == 0 when
%           i == j

N = size(x,1)/2;
R.avg = zeros(N,N);
R.var = zeros(N,N);
for i = 1:N
  for j = i+1:N
    d = norm(x(2*i-1:2*i) - x(2*j-1:2*j));
    [R.avg(i,j), R.var(i,j)] = channel(d);
  end
end
R.avg = R.avg + triu(R.avg,1)';
R.var = R.var + triu(R.var,1)';

end

% stochastic channel model (M1 in Jon Fink thesis)
function [rate, var] = channel(d)

% model parameters
L0 = -50.6; % dBm
n = 2.75;
sigma_F2 = 40.5; % variance of noise due to fading, ~N(0,sigma_F2)
N0 = -70; % noise at receiver (dBm)

% compute link rate
PR = L0 - 10*n*log10(d); % dBm
P = dBm2mW(PR); % mW
PN0 = dBm2mW(N0); % mW
rate = 1 - erfc(sqrt(P./PN0));

% compute link variance (using delta method approximation)
var = (log(10)/(10*sqrt(PN0*pi))*exp(-P/PN0).*10.^(PR/20)).^2*sigma_F2;

end

function mW = dBm2mW(dBm)

mW = 10.^(dBm/10);

end
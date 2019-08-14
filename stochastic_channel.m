function [rate, var] = stochastic_channel(d)

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
function cm = channel_params()
% CHANNEL_PARAMS a function returning the parameters of a stochastic channel
% model
%
% outputs:
%   cm.L0       - transmit power (dBm)
%   cm.n        - decay rate
%   cm.sigma_F2 - noise variance
%   cm.PN0      - noise at receiver (mW)

% from Jon Fink thesis page 104
cm = struct();
cm.L0 = -48;            % transmit power (dBm)
cm.PL0 = dBm2mW(cm.L0); % transmit power (mW)
cm.n = 2.52;            % decay rate
cm.sigma_F2 = 40;       % noise variance
N0 = -70;               % noise at receiver (dBm)
cm.PN0 = dBm2mW(N0);    % noise at receiver (mW)
cm.a = 0.2;             % sigmoid parameter 1
cm.b = 6.0;             % sigmoid parameter 2

% % model parameters (I chose)
% L0 = -50.6;      % dBm
% n = 2.75;        % decay rate
% sigma_F2 = 40.5; % variance of noise, ~N(0,sigma_F2)
% N0 = -70;        % noise at receiver (dBm)

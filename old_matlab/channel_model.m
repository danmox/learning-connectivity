function [rate, var] = channel_model(d)

cm = channel_params();

% compute link rate
PR = cm.L0 - 10*cm.n*log10(d); % dBm
P = dBm2mW(PR); % mW
rate = 1 - erfc(sqrt(P./cm.PN0));

% use sigmoid approximation to variance
var = (cm.a*d./(cm.b + d)).^2;

% % compute link variance (using delta method approximation)
% var = (log(10)/(10*sqrt(cm.PN0*pi))*exp(-P/cm.PN0).*10.^(PR/20)).^2*cm.sigma_F2;
% if isnan(var)
%   var = eps;
% end

end

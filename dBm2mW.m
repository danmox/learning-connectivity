function mW = dBm2mW(dBm)
% DBM2MW convert dBm to mW
%
% input:
%   dBm - Nx1 vector of dBm values to be converted
%
% output:
%   mW - Nx1 vector of values in milli-watts

mW = 10.^(dBm/10);

end
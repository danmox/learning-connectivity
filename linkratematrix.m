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
    [R.avg(i,j), R.var(i,j)] = stochastic_channel(d);
  end
end
R.avg = R.avg + triu(R.avg,1)';
R.var = R.var + triu(R.var,1)';

end
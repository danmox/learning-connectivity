function derivative = channel_derivative(xi, xj, wrt, type)

% difference between dRij/dxi and dRij/dxj is a sign
if wrt == 1     % with respect to 1st index (i.e. 'i')
  diff = xi - xj;
elseif wrt == 2 % with respect to 2nd index (i.e. 'j')
  diff = xj - xi;
else
  assert(0, 'unknown option wrt %d', wrt);
end

c = channel_params();
d = norm(diff);

if strcmp(type, 'var')
  dvar = log(10)^2 * c.PL0 * c.n * c.sigma_F2 / (100 * c.PN0^2 * pi)  * ...
    exp(-2 * c.PL0 * d^(-c.n) / c.PN0)  * ...
    (2 * c.PL0 / d^c.n  -  c.PN0)  * ...
    diff / d^(c.n + 2);
  dvar(isnan(dvar)) = 0;
  derivative = dvar;
elseif strcmp(type, 'mean')
  dmean = 2 * c.n * c.PL0 / (c.PN0 * sqrt(pi))  * ...
    exp(-c.PL0 * d^(-c.n) / c.PN0)  * ...
    diff / d^(c.n + 2);
  dmean(isnan(dmean)) = 0;
  derivative = dmean;
else
  assert(0, 'unknown option type %s', type);
end

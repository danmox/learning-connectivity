function [slack_var, routing_vars, status] = rrsocpmeanvar(x, qos, constrain_slack)
% RRSOCPMEANVAR solve the robust routing SOCP formulation that constrains
% the mean and variance of the node rate margins separately
%
% inputs:
%   x               - 2Nx1 column vector of [x;y] agent positions stacked
%   qos             - a Kx1 array of structs containing:
%                     flow.src   - the list of source nodes
%                     flow.dest  - the list of destination nodes
%                     margin     - the margin with which the constraint
%                                  must be met
%                     confidence - variance bound
%   constrain_slack - bool indicating if the following constraint should be
%                     added to the robust routing problem: slack >= 0
%
% outputs:
%   slack_var    - the slack variable from the LP solution
%   routing_vars - NxNxK matrix of routing variables
%   status       - the status of the LP; if isnan(status) == true then no
%                  feasible was found

N = size(x,1)/2;
K = length(qos);

%%  form LP coefficient matrices

R = linkratematrix(x);
[A,B,routing_var_mask] = nodemarginconsts(qos,R);

%% Solve SOCP

% node margin bounds
m_ik = zeros(N,K);
for k = 1:K
  m_ik(union(qos(k).flow.src, qos(k).flow.dest),k) = qos(k).margin;
end

% node variance bounds
v_ik = ones(N,K);
for k = 1:K
  v_ik(:,k) = v_ik(:,k) * sqrt(qos(k).confidence);
end

% slack bound
slack_bound = 0;
if ~constrain_slack
  slack_bound = -1e10; % sufficiently large number to be "unconstrained"
end

cvx_begin quiet
  variables routing_vars(N,N,K) slack_var
  y = [routing_vars(:); slack_var];
  expression mean_lhs(N,K)
  expression var_lhs(N,K)
  for k = 1:K
    for n = 1:N
      i = (k-1)*N + n;
      mean_lhs(n,k) = B(i,:) * y;
      var_lhs(n,k) = norm( diag( A(i,:) ) * y );
    end
  end
  maximize( slack_var )
  subject to
    mean_lhs >= m_ik
    var_lhs  <= v_ik
    0 <= routing_vars <= 1
    sum( sum(routing_vars, 3), 2) <= 1
    sum( sum(routing_vars, 3), 1) <= 1
    slack_var >= slack_bound
    routing_vars(routing_var_mask) == 0
cvx_end

status = ~isnan(slack_var); % check if a solution has been found
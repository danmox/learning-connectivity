function [slack_var, routing_vars, status] = rrsocpprobconf(x, qos, constrain_slack)
% RRSOCPPROBCONF solve the robust routing SOCP formulation that utilizes
% probability constraints on the network node margins
%
% inputs:
%   x               - 2Nx1 column vector of [x;y] agent positions stacked
%   qos             - a Kx1 array of structs containing:
%                     flow.src   - the list of source nodes
%                     flow.dest  - the list of destination nodes
%                     margin     - the margin with which the constraint
%                                  must be met
%                     confidence - probabilistic confidence that the
%                                  constraint holds
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

%%  form SOCP coefficient matrices

R = linkratematrix(x);
[A,B,zero_vars_mask] = nodemarginconsts(qos,R);

%% Solve SOCP

% confidence threshold
conf = norminv(vertcat(qos(:).confidence), 0, 1);

% node margins
m_ik = zeros(N,K);
for k = 1:K
  for i = 1:length(qos(k).flow.src)
    m_ik(qos(k).flow.src(i),k) = qos(:).margin;
  end
end

% slack bound
slack_bound = 0;
if ~constrain_slack
  slack_bound = -1e10; % sufficiently large number to be unconstrained
end

cvx_begin quiet
  variables routing_vars(N,N,K) slack_var
  y = [routing_vars(:); slack_var];
  expression lhs(N,K)
  expression rhs(N,K)
  for k = 1:K
    for n = 1:N
      i = (k-1)*N + n;
      lhs(n,k) = norm( diag( A(i,:) ) * y );
      rhs(n,k) = (B(i,:)*y - m_ik(n,k)) / conf(k);
    end
  end
  maximize( slack_var )
  subject to
    lhs <= rhs
    0 <= routing_vars <= 1
    sum( sum(routing_vars, 3), 2) <= 1
    sum( sum(routing_vars, 3), 1) <= 1
    slack_var >= slack_bound
    routing_vars(zero_vars_mask) == 0
cvx_end

status = ~isnan(slack_var); % check if a solution has been found
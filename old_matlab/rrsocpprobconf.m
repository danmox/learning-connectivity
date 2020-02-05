function [output, routing_vars, status] = rrsocpprobconf(x, qos, constrain_slack, e)
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

if nargin < 4
  e = 0.0;
end

N = size(x,1)/2;
K = length(qos);

%%  form SOCP coefficient matrices

R = linkratematrix(x);
[A,B,zero_vars_mask] = nodemarginconsts(qos,R);

%% Solve SOCP

% confidence threshold
conf = ones(N,K).*norminv(horzcat(qos(:).confidence) + e, 0, 1);
conf = reshape(conf, [N*K 1]);

% node margins
m_ik = zeros(N,K);
for k = 1:K
  m_ik([qos(k).flow.src qos(k).flow.dest],k) = qos(k).margin;
end
m_ik = reshape(m_ik, [N*K 1]);

% slack bound
slack_bound = 0;
if ~constrain_slack
  slack_bound = -1e2; % sufficiently large number to be unconstrained
end

num_const = size(A,1);
cvx_begin quiet
  variables routing_vars(N,N,K) slack_var
  y = [routing_vars(:); slack_var];
  expression lhs(num_const,1)
  expression rhs(num_const,1)
  expression min_margin
  for i = 1:num_const
    lhs(i) = norm(diag(A(i,:))*y);
    rhs(i) = (B(i,:)*y - m_ik(i)) / conf(i);
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

output = slack_var;

status = ~isnan(slack_var); % check if a solution has been found

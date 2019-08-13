function [slack_var, status] = robustroutingsocp(x, qos)
% ROBUSTROUTINGSOCP solve the robust routing problem without constraining the
% slack variable to be greater than 0. This should make the problem always
% feasible. When slack_var >= 0 then the network configuration is valid and
% when slack_var < 0 it is invalid. slack_var has the interpretation of
% being the margin with which the chance constraints are satisfied or not.
%
% inputs:
%   x   - 2Nx1 column vector of [x;y] agent positions stacked
%   qos - a Kx1 array of structs containing:
%         flow.src   - the list of source nodes
%         flow.dest  - the list of destination nodes
%         margin     - the margin with which the constraint must be met
%         confidence - probabilistic confidence that the constraint holds
%
% outputs:
%   slack_var - the slack variable from the SOCP solution
%   status    - the status of the SOCP; if isnan(status) == true then no
%               feasible was found

N = size(x,1)/2;
K = length(qos);

R = linkratematrix(x);

% form SOCP coefficient matrices
[A, B] = makesocp(qos, R);

% confidence threshold
conf = norminv(vertcat(qos(:).confidence), 0, 1);

% node margins
m_ik = vertcat(qos(:).margin);

% solve SOCP
cvx_begin quiet
  variables routing_vars(N,N,K) slack_var;
  y = [routing_vars(:); slack_var];
  expression lhs(K*N);
  for k = 1:N*K
    lhs(k) = norm( diag(A(k,:)) * y );
  end
  maximize( slack_var )
  subject to
    lhs <= (B*y - m_ik)./conf
    0 <= routing_vars <= 1
    sum( sum(routing_vars, 3), 2) <= 1
    slack_var >= 0
    routing_vars(logical(repmat(eye(N), [1 1 K]))) == 0
cvx_end

status = ~isnan(slack_var); % has a solution been found?
function rrsocpinfo(x, qos, routes, slack, verbosity)
% RRSOCPINFO print out easy to read information about the solution to a
% robust routing problem including: the constraint matrices used during
% optimization, channel info, optimal routing variables, constraints at
% optimality, and a visualization of the routes
%
% inputs:
%   x         - 2Nx1 vector of [x;y] node positions stacked
%   qos       - a Kx1 array of structs containing:
%               flow.src   - the list of source nodes
%               flow.dest  - the list of destination nodes
%               margin     - the margin with which the flow constraints
%                            must be met
%               confidence - variance bound
%   routes    - the optimal routing variables obtained by solving the
%               robust routing problem
%   slack     - the optimal slack
%   verbosity - a 1x5 boolean vector with elements determining what
%               information gets presented: [constraint_matrices,
%               channel_info, routing_variables, optimal_constraints,
%               plot_routes]

if nargin < 5
  verbosity = [1 1 1 1 1];
end

R = linkratematrix(x);
N = size(R.avg,1);
K = length(qos);
[A,B] = nodemarginconsts(qos,R);

if issparse(routes)
  routes = full(routes);
end

%% constraint matrices

if verbosity(1)

  % variable names
  var_names = cell(N,N,K);
  for k = 1:K
    for i = 1:N
      for j = 1:N
        var_names{i,j,k} = ['a_' num2str(i) num2str(j) '_' num2str(k)];
      end
    end
  end
  var_names = reshape(var_names, [1 N*N*K]);
  
  mask = logical([sum(abs(A(:,1:end-1))) ~= 0, 0]);

% contraint matrices
  Amat = array2table(round(A(:,find(mask)),3), 'VariableNames', var_names(mask(1:end-1)))
  Bmat = array2table(round(B(:,find(mask)),3), 'VariableNames', var_names(mask(1:end-1)))

end

%% channel info

if verbosity(2)
  % distance and channel information
  channel_info = table(round(squareform(pdist(reshape(x,[2 N])')),2),...
    round(R.avg,2),...
    round(R.var,2),...
    'VariableNames', {'distance','R_avg','R_var'})
end

%% optimal routing variables

if verbosity(3)
  % routing and channel usage
  solution_info = table(round(reshape(routes, [N N*K]),2),...
    round(sum(sum(routes,3),2),2),...
    round(sum(sum(routes,3),1),2)',...
    round(slack*ones(N,1),2),...
    'VariableNames', {'routes','Tx_usage','Rx_usage','slack'})
end

%% constraints at optimality

if verbosity(4)
  % node margin bounds
  m_ik = zeros(N,K);
  for k = 1:K
    m_ik(union(qos(k).flow.src, qos(k).flow.dest),k) = qos(k).margin;
  end
  
  % constraints
  [A,B] = nodemarginconsts(qos, linkratematrix(x));
  y = [routes(:); 0];
  mean_lhs = zeros(N,K);
  var_lhs = zeros(N,K);
  for k = 1:K
    for n = 1:N
      i = (k-1)*N + n;
      mean_lhs(n,k) = B(i,:)*y - m_ik(n,k);
      var_lhs(n,k) = norm(diag(A(i,:))*y);
    end
  end
  table(mean_lhs, slack*ones(N,K), var_lhs, sqrt(horzcat(qos(:).confidence).*ones(N,K)),...
    'VariableNames', {'mean_lhs', 'slack', 'var_lhs', 'confidence'})
end
    
%% plot routes

if verbosity(5)
  % plot results
  plotroutes(gcf, x, routes, qos)
end
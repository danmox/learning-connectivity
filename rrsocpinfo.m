function rrsocpinfo(x, qos, routes, slack, socp_solution)

if nargin < 5
  socp_solution = false;
end

R = linkratematrix(x);
N = size(R.avg,1);
K = length(qos);
[A,B] = nodemarginconsts(qos,R);

if issparse(routes)
  routes = full(routes);
end

var_names = cell(N,N,K);
for k = 1:K
  for i = 1:N
    for j = 1:N
      var_names{i,j,k} = ['a_' num2str(i) num2str(j) '_' num2str(k)];
    end
  end
end

% contraint matrices
Amat = array2table(A(:,1:end-1),'VariableNames', reshape(var_names, [1 N*N*K]))
Bmat = array2table(B(:,1:end-1),'VariableNames', reshape(var_names, [1 N*N*K]))

% distance and channel information
table(round(squareform(pdist(reshape(x,[2 N])')),3),...
      round(R.avg,3),...
      round(R.var,3),...
      'VariableNames', {'distance','R_avg','R_var'})

% routing and channel usage
table(round(reshape(routes, [N N*K]),3),...
      round(sum(sum(routes,3),2),2),...
      round(sum(sum(routes,3),1),2)',...
      'VariableNames', {'routes','Tx_usage','Rx_usage'})

table(slack,'VariableNames', {'slack'})

% % node margins
% m_ik = zeros(N,K);
% for k = 1:K
%   for i = 1:length(qos(k).flow.src)
%     m_ik(qos(k).flow.src(i),k) = qos(:).margin;
%   end
% end
% 
% % confidence threshold
% conf = norminv(vertcat(qos(:).confidence), 0, 1);
% 
% % chance constraints
% y = [routes(:); 0]; % no slack in v(alpha,x)
% v_alpha_x = zeros(N,K);
% if socp_solution
%   for k = 1:K
%     for n = 1:N
%       v_alpha_x(n,k) = (B((k-1)*N+n,:)*y - m_ik(n,k)) / norm( diag( A((k-1)*N+n,:) ) * y ) - conf(k);
%     end
%   end
% else
%   for k = 1:K
%     for n = 1:N
%       v_alpha_x(n,k) = (B((k-1)*N+n,:)*y - m_ik(n,k)) / norm( diag( A((k-1)*N+n,:) ) * y ) - conf(k);
%     end
%   end
% end
% table(v_alpha_x, slack*ones(N,1), 'VariableNames', {'v_alpha_x','slack'})

% plot results
plotroutes(gcf, x, routes, qos)
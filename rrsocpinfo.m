function rrsocpinfo(x, qos, routes, slack, verbosity)

if nargin < 5
  verbosity = 4;
end

print_const_mats = true;
print_channel_info = true;
create_figures = true;
switch verbosity
  case 1 % print routes
    print_const_mats = false;
    print_channel_info = false;
    create_figures = false;
  case 2 % print routes and create figure
    print_const_mats = false;
    print_channel_info = false;
  case 3 % pring routes and channel info and create figure
    print_const_mats = false;
end % otherwise: print everything and create the figures

R = linkratematrix(x);
N = size(R.avg,1);
K = length(qos);
[A,B] = nodemarginconsts(qos,R);

if issparse(routes)
  routes = full(routes);
end

if print_const_mats

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

if print_channel_info
  % distance and channel information
  channel_info = table(round(squareform(pdist(reshape(x,[2 N])')),2),...
    round(R.avg,2),...
    round(R.var,2),...
    'VariableNames', {'distance','R_avg','R_var'})
end

% routing and channel usage
solution_info = table(round(reshape(routes, [N N*K]),2),...
      round(sum(sum(routes,3),2),2),...
      round(sum(sum(routes,3),1),2)',...
      round(slack*ones(N,1),2),...
      'VariableNames', {'routes','Tx_usage','Rx_usage','slack'})

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
if create_figures
  plotroutes(gcf, x, routes, qos)
end
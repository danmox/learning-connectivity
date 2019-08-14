%% initialize parameters
clc;clear;

% communication requirements, agent: 2 -> 1
qos(1) = struct('flow', struct('src', 2, 'dest', 1),...
                'margin', 0.2,...
                'confidence', 0.7);
              
% fixed task agent locations
dist = 15;
x_task = [0 0 dist 0]';

% indexing
task_agent_count = 2;
comm_agent_count = 1;
It = 1:2*task_agent_count; % task_agent_count [x;y] pairs stacked
Itx = It(1:2:end);
Ity = It(2:2:end);
Ic = (1:2*comm_agent_count) + 2*task_agent_count; % comm_agent_count [x;y] pairs stacked
Icx = Ic(1:2:end);
Icy = Ic(2:2:end);

% distribute comm agents about circle (initial config)
x_comm = [dist/2 0];

% team config
x = zeros(2*(task_agent_count + comm_agent_count), 1);
x(It) = x_task;
x(Ic) = x_comm;

%% solve routing problem

[slack, routes, status] = robustroutingsocp(x, qos, true);

%% analysis

R = linkratematrix(x);
N = size(R.avg,1);
K = length(qos);
table(round(squareform(pdist(reshape(x,[2 3])')),3),...
      round(R.avg,3),...
      round(R.var,3),...
      round(reshape(routes, [N N*K]),3),...
      round(sum(sum(routes,3),2),2),...
      round(sum(sum(routes,3),1),2)',...
      'VariableNames', {'distance','R_avg','R_var','routes','Tx_usage', 'Rx_usage'})

% node margins
m_ik = zeros(N,K);
for k = 1:K
  for i = 1:length(qos(k).flow.src)
    m_ik(qos(k).flow.src(i),k) = qos(:).margin;
  end
end

% confidence threshold
conf = norminv(vertcat(qos(:).confidence), 0, 1);

% chance constraints
[A,B] = makesocpconsts(qos,R);
y = [routes(:); slack];
lhs = zeros(N,K);
rhs = zeros(N,K);
for k = 1:K
  for n = 1:N
    lhs(n,k) = norm( diag( A((k-1)*N+n,:) ) * y );
    rhs(n,k) = (B((k-1)*N+n,:)*y - m_ik(n,k)) / conf(k);
  end
end
table(lhs,rhs,slack*ones(N,1),'VariableNames',{'lhs','rhs','slack'})

% plot results
ax = gca;
x_plot = x;
x_plot(6) = x_plot(6)+1;
plotroutes(ax, x_plot,routes)
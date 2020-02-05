%% parameters
clc;clear;

net_x0=4.0;
net_y0=4.0;
net_xf=23.0;
net_yf=11.0;
ym = mean([net_y0, net_yf]);
xm = mean([net_x0, net_xf]);

sample_count = 1000;
sample_variance = 0.3;
x_task = [[net_x0;net_y0], [0;ym], [net_xf;net_yf], [8;6]];
% x_comm = [[xm-3.0;ym], [xm+3.0;ym]];
x_comm = [6;6];

% communication requirements
qos(1) = struct('flow', struct('src', 1, 'dest', 2),...
                'margin', 0.15,...
                'confidence', 0.8);
qos(2) = struct('flow', struct('src', 2, 'dest', 1),...
                'margin', 0.15,...
                'confidence', 0.8);
qos(3) = struct('flow', struct('src', 3, 'dest', 2),...
                'margin', 0.15,...
                'confidence', 0.8);
qos(4) = struct('flow', struct('src', 2, 'dest', 3),...
                'margin', 0.15,...
                'confidence', 0.8);

% indexing
task_agent_count = size(x_task,2);
comm_agent_count = size(x_comm,2);
It = 1:task_agent_count;
Ic = (1:comm_agent_count) + task_agent_count;

% team config
x = [x_task, x_comm];
x_star = x;

N = size(x,2);
K = length(qos);

% confidence threshold
conf = ones(N,K).*norminv(horzcat(qos(:).confidence), 0, 1);
conf = reshape(conf, [N*K 1]);

% node margins
m_ik = zeros(N,K);
for k = 1:K
  m_ik([qos(k).flow.src qos(k).flow.dest],k) = qos(k).margin;
end
m_ik = reshape(m_ik, [N*K 1]);

% initialize slack figure
figure(2); clf; hold on;
slack_ax = gca;
% slack_ax.YLim = [0.0 0.05];
% slack_ax.YLim = [0.03 0.06];

idx = 1;
slack0 = 0;

%% network reconfiguration
its = 100;
for it = 1:its
  
  fprintf('\niteration %d\n', it);
  
  % find optimal routing variables
  [slack, routes, ~] = rrsocpprobconf(x_star(:), qos, true, 0.05);
  fprintf('slack = %.4f\n', slack);
  routes = full(routes);
  if idx == 1
    plot(slack_ax, idx, slack, 'bx', 'MarkerSize', 10, 'LineWidth', 2);
  else
    plot(slack_ax, [idx-1 idx], [slack0 slack], 'b', 'LineWidth', 2);
  end
  drawnow
  slack0 = slack;
  slack_star = slack;
  idx = idx + 1;
  
  % update network configuration
  x0 = x_star;
  sample_hist = zeros(2,comm_agent_count,sample_count);
  slack_updates = 0;
  for i = 1:sample_count
    
    x_prime = x0;
    x_prime(:,Ic) = x_prime(:,Ic) + sample_variance*randn(2,comm_agent_count);
    sample_hist(:,:,i) = x_prime(:,Ic);
    
    % compute slack
    
    slack_prime = min(computeSlack(qos, x_prime, m_ik, conf, routes));
    
    % update best config
    
    if slack_prime - slack_star > 0.001
      slack_updates = slack_updates + 1;
      slack_star = slack_prime;
      x_star(:,Ic) = x_prime(:,Ic);
    end
    
  end
  fprintf('slack_star updated %d times\n', slack_updates);
  
  figure(1); clf; hold on;
  config_ax = gca;
  for i = 1:comm_agent_count
    plot(reshape(sample_hist(1,i,:), [sample_count 1]),...
      reshape(sample_hist(2,i,:), [sample_count 1]), '.', 'Color', [1 0.8 0.8]);
  end
  plot(x_star(1,It), x_star(2,It), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
  plot(x0(1,Ic), x0(2,Ic), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
  plot(config_ax, x_star(1,Ic), x_star(2,Ic), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
  axis equal
  drawnow
  
%   % analysis
%   [slack_star, routes_star] = rrsocpprobconf(x_star(:), qos, true);
%   if slack_star < slack
%     
%     s0 = computeSlack(qos, x0, m_ik, conf, routes);
%     s = computeSlack(qos, x_star, m_ik, conf, routes);
%     s_star = computeSlack(qos, x_star, m_ik, conf, routes_star);
%     fprintf('slack0 = %.4f, slack_star = %.4f\n', slack, slack_star);
%     table(round(s0,4), round(s,4), round(s_star,4), 'VariableNames', {'s0','s','s_star'})
%   end
  
end

%% helper functions

function s = computeSlack(qos, x, m_ik, conf, routes)

% active constraints
R = linkratematrix(x(:));
[A,B,~] = nodemarginconsts(qos,R);

num_const = size(A,1);
s = zeros(num_const,1);
y = [routes(:); 0];
for j = 1:num_const
  s(j) = B(j,:)*y - m_ik(j) - conf(j)*norm(diag(A(j,:))*y);
end

end

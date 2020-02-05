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
x_comm = [5;12];

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
slack_ax.YLim = [0.03 0.06];

idx = 1;
slack0 = 0;

%% network reconfiguration
its = 80;
for it = 1:its
  
  fprintf('\niteration %d\n', idx+1);
  
  % find optimal routing variables
  [slack, routes, ~] = rrsocpprobconf(x_star(:), qos, true);
  fprintf('slack = %.4f\n', slack);
  routes = full(routes);
  if idx == 1
    plot(slack_ax, idx, slack, 'bx', 'MarkerSize', 10, 'LineWidth', 2);
  else
    plot(slack_ax, [idx-1 idx], [slack0 slack], 'b', 'LineWidth', 2);
  end
  slack0 = slack;
  idx = idx + 1;
  
  % compute active margin constraints
  s = computeSlack(qos, x_star, m_ik, conf, routes);
  % active = abs(s - slack) < 1e-6;
  active = true(size(s));
  fprintf('%d active margin constraints\n', nnz(active));
  if nnz(active) == 0
    disp('no active margin constraints');
  end
  
  % compute vax_star
  v = computeV(qos, x_star, m_ik, conf, routes);
  vax_star = min(v(active));
  
  % update network configuration
  x0 = x_star;
  sample_hist = zeros(2,comm_agent_count,sample_count);
  v_updates = 0;
  for i = 1:sample_count
    
    x_prime = x0;
    x_prime(:,Ic) = x_prime(:,Ic) + sample_variance*randn(2,comm_agent_count);
    sample_hist(:,:,i) = x_prime(:,Ic);
    
    % compute v(a,x)
    
    v = computeV(qos, x_prime, m_ik, conf, routes);
    
    if nnz(active) == 0
      disp('no active margin constraints');
      continue
    end
    
    % update best config
    
    vax_prime = min(v(active));
    if vax_prime > vax_star
      v_updates = v_updates + 1;
      vax_star = vax_prime;
      x_star(:,Ic) = x_prime(:,Ic);
    end
    
  end
  fprintf('vax_star updated %d times\n', v_updates);
  
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
  
  % analysis
  [slack_star, routes_star] = rrsocpprobconf(x_star(:), qos, true);
  if slack_star < slack
    
    s0 = computeSlack(qos, x0, m_ik, conf, routes);
    v0 = computeV(qos, x0, m_ik, conf, routes);
    s = computeSlack(qos, x_star, m_ik, conf, routes);
    v = computeV(qos, x_star, m_ik, conf, routes);
    s_star = computeSlack(qos, x_star, m_ik, conf, routes_star);
    %   active_star = abs(s_star - slack_star) < 1e-6;
    active_star = true(size(s_star));
    v_star = computeV(qos, x_star, m_ik, conf, routes_star);
    fprintf('slack0 = %.4f, slack_star = %.4f\n', slack, slack_star);
    fprintf('min(v0(active)) = %.4f\n', min(v0(active)));
    fprintf('min(v(active)) = %.4f\n', min(v(active)));
    fprintf('min(v_star(active_star)) = %.4f\n', min(v_star(active_star)));
    table(active, round(s0,4), round(v0,4), round(s,4), round(v,4),...
      active_star, round(s_star,4), round(v_star,4),...
      'VariableNames', {'active','s0','v0','s','v','active_star','s_star', 'v_star'})
    %   warning('resetting configuration');
    %   x_star = x0;
  end
  
end

%% helper functions

function v = computeV(qos, x, m_ik, conf, routes)

R = linkratematrix(x(:));
[A,B,~] = nodemarginconsts(qos,R);
y = [routes(:); 0];

num_const = size(A,1);
v = zeros(num_const,1);
for j = 1:num_const
  v(j) = (B(j,:)*y - m_ik(j)) / norm(diag(A(j,:))*y) - conf(j);
end

end

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

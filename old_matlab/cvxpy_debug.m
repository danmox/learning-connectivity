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

% find optimal routing variables
[slack, routes, ~] = rrsocpprobconf(x_star(:), qos, true);
fprintf('slack = %.4f\n', slack);
routes = full(routes);

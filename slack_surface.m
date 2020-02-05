%% parameters
clc;clear;

net_x0=4.0;
net_y0=4.0;
net_xf=23.0;
net_yf=11.0;
ym = mean([net_y0, net_yf]);
xm = mean([net_x0, net_xf]);

samples = 50;
x_task = [[net_x0;net_y0], [0;ym], [net_xf;net_yf], [8;6]];
x_comm = [12;8];

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

%% slack surface for fixed alpha

dx = 2.0;
[x_pts, y_pts] = meshgrid(linspace(x_comm(1)-dx, x_comm(1)+dx, samples),...
                          linspace(x_comm(2)-dx, x_comm(2)+dx, samples));            
x_pts = x_pts(:); y_pts = y_pts(:);

[s, routes, ~] = rrsocpprobconf(x(:), qos, true, 0.05);
s = s
s_true = rrsocpprobconf(x(:), qos, true)

slack = zeros(size(x_pts));
for i = 1:length(x_pts)
  x(:,Ic) = [x_pts(i); y_pts(i)];
  slack(i) = min(computeSlack(qos, x, m_ik, conf, routes));
end
[max_slack, max_idx] = max(slack);
max_slack = max_slack

% sanity check
x(:,Ic) = [x_pts(max_idx); y_pts(max_idx)];
s_new = rrsocpprobconf(x(:), qos, true)

%% plots

figure(1);clf;hold on;
surf(reshape(x_pts, samples*[1 1]), reshape(y_pts, samples*[1 1]), reshape(slack, samples*[1 1]))
plot3(x_comm(1), x_comm(2), s, 'rx', 'MarkerSize', 15, 'LineWidth', 2)
plot3(x_pts(max_idx), y_pts(max_idx), slack(max_idx), 'bx', 'MarkerSize',10, 'LineWidth', 2)

figure(2);clf;hold on;
plot(x_task(1,:), x_task(2,:), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(x_comm(1,:), x_comm(2,:), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
plot(x_pts(max_idx), y_pts(max_idx), 'gx', 'MarkerSize',10, 'LineWidth', 2)
text(x_task(1,:)+0.5, x_task(2,:)+0.5, vec2cellstr(It), 'FontSize', 12, 'FontWeight', 'bold')
text(x_comm(1,:)+0.5, x_comm(2,:)+0.5, vec2cellstr(Ic), 'FontSize', 12, 'FontWeight', 'bold')
axis equal

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

function char_vec = vec2cellstr(vec)

char_vec = cell(1,length(vec));
for i = 1:length(vec)
  char_vec{i} = num2str(vec(i));
end

end
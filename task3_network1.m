%% test CVX for 3 task 6 network agent case

clear; clc;

patrol_rad = 20;
comm_rad = 8;
task_agent_count = 3;
comm_agent_count = 6;

margin = 0.02;
confidence = 0.7;
qos(1) = struct('flow', struct('src', 1, 'dest', [2 3]),...
                'margin', margin, 'confidence', confidence);
qos(2) = struct('flow', struct('src', 2, 'dest', [1 3]),...
                'margin', margin, 'confidence', confidence);
qos(3) = struct('flow', struct('src', 3, 'dest', [1 2]),...
                'margin', margin, 'confidence', confidence);

x_task = [patrol_rad*cos(2*pi/task_agent_count.*(0:task_agent_count-1));
          patrol_rad*sin(2*pi/task_agent_count.*(0:task_agent_count-1))];
x_comm = [5;5];
x = [x_task x_comm];

figure(1);clf;hold on;
plot(x_task(1,:), x_task(2,:), 'rx', 'MarkerSize', 16, 'LineWidth', 2)
plot(x_comm(1,:), x_comm(2,:), 'bo', 'MarkerSize', 16, 'LineWidth', 2)
axis equal

%% solve SOCP

% for i = 1:10
  tic
  [slack, routing_vars, status] = rrsocpprobconf(x(:), qos, true);
%   x = x + randn(size(x));
  toc
% end

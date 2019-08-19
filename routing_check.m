%% SOCP formulation
clc;clear;

% communication requirements, agent: 2 -> 1
qos_socp(1) = struct('flow', struct('src', 2, 'dest', 1),...
                     'margin', 0.1,...
                     'confidence', 0.8);
% qos_socp(2) = struct('flow', struct('src', 1, 'dest', 2),...
%                      'margin', 0.1,...
%                      'confidence', 0.8);
              
% fixed task agent locations
dist = 15;
x_task = [0 0 dist 0]';

% distribute comm agents about circle (initial config)
x_comm = [1/3*dist 1 2/3*dist -1];

% indexing
task_agent_count = length(x_task)/2;
comm_agent_count = length(x_comm)/2;
It = 1:2*task_agent_count; % task_agent_count [x;y] pairs stacked
Itx = It(1:2:end);
Ity = It(2:2:end);
Ic = (1:2*comm_agent_count) + 2*task_agent_count; % comm_agent_count [x;y] pairs stacked
Icx = Ic(1:2:end);
Icy = Ic(2:2:end);

% team config
x = zeros(2*(task_agent_count + comm_agent_count), 1);
x(It) = x_task;
x(Ic) = x_comm;

% solve routing problem

[slack, routes, ~] = robustroutingsocp(x, qos_socp, true);

% analysis
rrsocpinfo(x,qos_socp,routes,slack)
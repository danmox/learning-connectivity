%% Team Configuration

clc;clear;

dist = 15;                           % distance between task agents
x_task = [0 0 dist 0]';              % task agent locations
x_comm = [8 5]';                     % network agent locations

% team config
x = [x_task; x_comm];

%% SOCP confidence formulation (constrained slack)

% communication requirements, agent: 2 -> 1
qos_socp(1) = struct('flow', struct('src', 2, 'dest', 1),...
                     'margin', 0.05,...
                     'confidence', 0.98);
% qos_socp(1) = struct('flow', struct('src', 2, 'dest', 1),...
%                      'margin', 0.2,...
%                      'confidence', 0.8);

% solve routing problem
constrain_slack = false;
[slack, routes, ~] = rrsocpprobconf(x, qos_socp, true);

% analysis
figure(1);clf;
rrsocpinfo(x,qos_socp,routes,slack,[1 1 1 1 1])
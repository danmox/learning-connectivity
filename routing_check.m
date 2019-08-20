%% Team Configuration

clc;clear;

constrain_slack = true;              % enforce slack >= 0 during optimization
dist = 15;                           % distance between task agents
x_task = [0 0 dist 0]';              % task agent locations
x_comm = [dist/2 1]';                % network agent locations

% team config
x = [x_task; x_comm];

%% SOCP confidence formulation

% communication requirements, agent: 2 -> 1
qos_socp(1) = struct('flow', struct('src', 2, 'dest', 1),...
                     'margin', 0.1,...
                     'confidence', 0.8);

% solve routing problem
[slack, routes, ~] = rrsocpprobconf(x, qos_socp, constrain_slack);

% analysis
figure(1);clf;
rrsocpinfo(x,qos_socp,routes,slack)
title('probabilistic confidence');

%% SOCP mean/var formulation

% communication requirements, agent: 2 -> 1
qos_socp(1) = struct('flow', struct('src', 2, 'dest', 1),...
                     'margin', 0.2,...
                     'confidence', 0.1);

% solve routing problem
[slack, routes, ~] = rrsocpmeanvar(x, qos_socp, constrain_slack);

% analysis
figure(2);clf;
rrsocpinfo(x,qos_socp,routes,slack)
title('mean / variance');
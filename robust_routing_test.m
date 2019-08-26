% this is a script for testing the effect of different qos parameters on
% the routes produced by the robust routing optimization problem as well as
% testing different robust routing formulations

%% Team Configuration

clc;clear;

dist = 15;                           % distance between task agents
x_task = [0 0 dist 0]';              % task agent locations
x_comm = [dist/3 1 2*dist/3 -1]';    % network agent locations

% team config
x = [x_task; x_comm];

%% SOCP mean/var formulation

% communication requirements, agent: 2 -> 1
qos_socp(1) = struct('flow', struct('src', 2, 'dest', 1),...
                     'margin', 0.3,...
                     'confidence', 0.2);
% qos_socp(2) = struct('flow', struct('src', 1, 'dest', 2),...
%                      'margin', 0.2,...
%                      'confidence', 0.1);

% % solve unconstrained-slack routing problem
% disp('unconstrained');
% [slack, routes, ~] = rrsocpmeanvar(x, qos_socp, false);
% 
% % analysis
% figure(1);clf;
% rrsocpinfo(x, qos_socp, routes,slack, 3)
% title('unconstrained');

% solve constrained-slack routing problem
disp('constrained');
[slack, routes, ~] = rrsocpmeanvar(x, qos_socp, true);

% analysis
figure(2);clf;
rrsocpinfo(x, qos_socp, routes, slack, 2)
% title('constrained');
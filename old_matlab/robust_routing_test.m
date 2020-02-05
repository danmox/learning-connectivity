% this is a script for testing the effect of different qos parameters on
% the routes produced by the robust routing optimization problem as well as
% testing different robust routing formulations

%% Team Configuration

clc;clear;

dist = 30;                              % distance between task agents
x_task = [0 0 dist 0]';                 % task agent locations
x_comm = [dist/2-5 2.0 dist/2+5 -2.0]'; % network agent locations
formulation = 'confidence';             % robust routing formulation

if strcmp(formulation, 'confidence')
  optprob = @rrsocpprobconf;
end
if strcmp(formulation, 'meanvar')
  optprob = @rrsocpmeanvar;
end

% team config
x = [x_task; x_comm];

%% solve SOCP

% communication requirements, agent: 2 -> 1
if strcmp(formulation, 'confidence')
  qos_socp(1) = struct('flow', struct('src', 2, 'dest', 1),...
    'margin', 0.1,...
    'confidence', 0.7);
end
if strcmp(formulation, 'meanvar')
  qos_socp(1) = struct('flow', struct('src', 2, 'dest', 1),...
    'margin', 0.01,...
    'confidence', 0.001);
end

% solve constrained-slack routing problem
disp(formulation);
[slack, routes, ~] = optprob(x, qos_socp, true);

% analysis
figure(2);clf;
rrsocpinfo(x, qos_socp, routes, slack, [1 1 1 0 1])
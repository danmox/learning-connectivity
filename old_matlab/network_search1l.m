% this script performs exhaustive search in 1 dimension for the optimal
% network configuration:
% 
% 0                       dist
% |-------------------------|
% O<-------O<---------------O
% x1       x3               x2
% |------->|
%
%% parameters
clc;clear;

rrtype = 'meanvar';           % options: meanvar, confidence
constrain_slack = true;       % enforce s >= 0
config_visualization = false; % draw the configs as they are checked
sample_count = 100;           % discretization degree
dist = 13;                    % distance between task agents
x_task = [[0;0], [dist;0]];  % task team locations
x_comm = zeros(2,1);          % network team starting configuration

rroptimization = @rrsocpprobconf;

% communication requirements, agent: 2 -> 1
qos(1) = struct('flow', struct('src', 2, 'dest', 1),...
  'margin', 0.3,...   % min rate margin
  'confidence', 0.1); % variance bound

% indexing
task_agent_count = size(x_task,2);
comm_agent_count = size(x_comm,2);
It = [1 2];
Ic = [3];

%% parameter search

% form angle and stretch parameter search space
x3 = linspace(0, dist, sample_count);
slack = zeros(size(x3)); % slack of resulting network

if config_visualization
  figure(1);clf; hold on;
  axis equal;
end

h = waitbar(0, 'Performing parameter search');
for i = 1:length(x3)
    
  % new team config
  x_new = make_config(x_task, x3(i));
  
  % find unconstrained slack
  [slack(i), ~, ~] = rroptimization(x_new(:), qos, constrain_slack);
  
  if config_visualization
    plot(x_new(1,It), x_new(2,It), 'r.', 'MarkerSize', 30);
    plot(x_new(1,Ic), x_new(2,Ic), 'b.', 'MarkerSize', 30);
    drawnow
  end
  
  waitbar(i/length(x3),h);
end
close(h)

% a best configuration
[~, max_idx] = max(slack);

%% slack surface visualization

figure(2);clf;hold on;
plot(x3, slack, 'k', 'LineWidth', 2.0);
plot(x3(max_idx), slack(max_idx), 'r.', 'MarkerSize', 30);
xlabel('$x_3$', 'Interpreter', 'latex', 'FontSize', 16)
ylabel('slack', 'Interpreter', 'latex', 'FontSize', 16)

%% one best configuration

clc;
x_star = make_config(x_task, x3(max_idx));

figure(3);clf;
[s, routes, status] = rroptimization(x_star(:), qos, constrain_slack);
rrsocpinfo(x_star(:), qos, routes, s);

%% debug configs

% verbosity = 3;
% 
% clc;
% disp('somehow not best solution');
% test_config(qos, rroptimization, x_task, 13/3, 2*13/3, 3);
% disp('best solution');
% test_config(qos, rroptimization, x_task, x1(max_idx), x2(max_idx), 3);

%% helper functions

function x = make_config(x_task, x1x)

x = [x_task, [x1x;0]];

end

function test_config(qos, optfunc, x_task, x3, verbosity)

% new team config
x_new = make_config(x_task, x3);

% find unconstrained slack
[slack, routes, ~] = optfunc(x_new, qos, true);

% info
figure(4);
rrsocpinfo(x_new, qos, routes, slack);

end

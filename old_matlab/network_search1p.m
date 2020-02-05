% this script performs exhaustive search in 2 dimensions for the optimal
% network configuration:
% 
% 0                       dist
% |-------------------------|
%
% ^            O
% |
% O<------------------------O
% x1                        x2
% |----------->|
%              x3

%% parameters
clc;clear;

constrain_slack = true;       % enforce slack >= 0 during optimization
config_visualization = false; % draw the configs as they are checked
sample_count = 21;            % discretization degree
dist = 15;                    % distance between task agents
x_task = [[0;0], [dist;0]];   % task team locations
x_comm = zeros(2,1);          % network team starting configuration

disp('using probabilistic confidence formulation');
rroptimization = @rrsocpprobconf;

% communication requirements, agent: 2 -> 1
qos(1) = struct('flow', struct('src', 2, 'dest', 1),...
  'margin', 0.25,...   % rate margin min
  'confidence', 0.90);  % probabilistic confidence

% indexing
task_agent_count = size(x_task,2);
comm_agent_count = size(x_comm,2);
It = 1:task_agent_count;
Ic = (1:comm_agent_count) + task_agent_count;

%% parameter search

% parameter search space
padding = dist*0.2;
xspace = linspace(0-padding, dist+padding, sample_count);
yspace = xspace - xspace(ceil(length(xspace)/2));
[x, y] = meshgrid(xspace, yspace);
x = x(:);
y = y(:);
slack = zeros(size(x)); % slack of resulting network

if config_visualization
  figure(1);clf; hold on;
  axis equal;
end

h = waitbar(0, 'Performing parameter search');
for i = 1:length(x)
    
  % new team config
  x_new = make_config(x_task, x(i), y(i));
  
  % find unconstrained slack
  [slack(i), ~, ~] = rroptimization(x_new(:), qos, constrain_slack);
  
  if config_visualization
    plot(x_new(:,It), x_new(:,It), 'r.', 'MarkerSize', 30);
    plot(x_new(:,Ic), x_new(:,Ic), 'b.', 'MarkerSize', 30);
    drawnow
  end
  
  waitbar(i/length(x),h);
end
close(h)

% a best configuration
[~, max_idx] = max(slack);

%% slack surface visualization

x_viz = reshape(x, sample_count*[1 1]);
y_viz = reshape(y, sample_count*[1 1]);
slack_viz = reshape(slack, sample_count*[1 1]);

figure(2);clf;hold on;
surf(x_viz, y_viz, slack_viz, 'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', 0.8)
grid on
contour3(x_viz, y_viz, slack_viz, 40, 'Color', 'k', 'LineWidth', 1);
% contour3(x3_viz, x4_viz, slack_viz,...
%   linspace(slack(max_idx)*0.99, slack(max_idx), 10),...
%   'Color', 'r', 'LineWidth', 1);
plot3(x(max_idx), y(max_idx), slack(max_idx), 'r.', 'MarkerSize', 30);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 18)
h = get(gca,'DataAspectRatio');
set(gca,'DataAspectRatio', [1 1 1/h(1)]);

%% one best configuration

clc;
x_star = make_config(x_task, x(max_idx), y(max_idx));

figure(3);clf;
[s, routes, status] = rroptimization(x_star(:), qos, constrain_slack);
rrsocpinfo(x_star(:), qos, routes, s, [0 0 1 0 1]);

%% debug configs

% verbosity = 3;
% 
% clc;
% disp('somehow not best solution');
% test_config(qos, rroptimization, x_task, 13/3, 2*13/3, 3);
% disp('best solution');
% test_config(qos, rroptimization, x_task, x1(max_idx), x2(max_idx), 3);

%% helper functions

function x = make_config(x_task, x, y)

x = [x_task, [x; y]];

end

function test_config(qos, optfunc, x_task, x1, x2, verbosity)

% new team config
x_new = make_config(x_task, x1, x2);

% find unconstrained slack
[slack, routes, ~] = optfunc(x_new, qos, true);

figure(4);
rrsocpinfo(x_new, qos, routes, slack);

end

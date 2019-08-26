% this script performs exhaustive search 2 dimensions of the network
% configuration

%% parameters
clc;clear;

rrtype = 'meanvar';        % options: meanvar, confidence
constrain_slack = true;       % enforce slack >= 0 during optimization
config_visualization = false; % draw the configs as they are checked
sample_count = 20;            % discretization degree
dist = 13;                    % distance between task agents
x_task = [0 0 dist 0]';       % task team locations
x_comm = zeros(4,1);          % network team starting configuration

if strcmp(rrtype,'meanvar')
  disp('using mean/var formulation');
  rroptimization = @rrsocpmeanvar;

  % communication requirements, agent: 2 -> 1
  qos(1) = struct('flow', struct('src', 2, 'dest', 1),...
    'margin', 0.3,...   % min rate margin
    'confidence', 0.1); % variance bound
else
  disp('using probabilistic confidence formulation');
  rroptimization = @rrsocpprobconf;

  % communication requirements, agent: 2 -> 1
  qos(1) = struct('flow', struct('src', 2, 'dest', 1),...
    'margin', 0.1,...   % rate margin min
    'confidence', 0.8); % probabilistic confidence
end

% indexing
task_agent_count = numel(x_task)/2;
comm_agent_count = numel(x_comm)/2;
It = 1:2*task_agent_count; % task_agent_count [x;y] pairs stacked
Itx = It(1:2:end);
Ity = It(2:2:end);
Ic = (1:2*comm_agent_count) + 2*task_agent_count; % comm_agent_count [x;y] pairs stacked
Icx = Ic(1:2:end);
Icy = Ic(2:2:end);

%% parameter search

% form angle and stretch parameter search space
xspace = linspace(0, dist, sample_count);
[x3, x4] = meshgrid(xspace, xspace);
x3 = x3(:);
x4 = x4(:);
slack = zeros(size(x3)); % slack of resulting network

if config_visualization
  figure(1);clf; hold on;
  axis equal;
end

h = waitbar(0, 'Performing parameter search');
for i = 1:length(x3)
    
  % new team config
  x_new = make_config(x_task, x3(i), x4(i));
  
  % find unconstrained slack
  [slack(i), ~, ~] = rroptimization(x_new, qos, constrain_slack);
  
  if config_visualization
    plot(x_new(Itx), x_new(Ity), 'r.', 'MarkerSize', 30);
    plot(x_new(Icx), x_new(Icy), 'b.', 'MarkerSize', 30);
    drawnow
  end
  
  waitbar(i/length(x3),h);
end
close(h)

% a best configuration
[~, max_idx] = max(slack);

%% slack surface visualization

x1_viz = reshape(x3, sample_count*[1 1]);
x2_viz = reshape(x4, sample_count*[1 1]);
slack_viz = reshape(slack, sample_count*[1 1]);

figure(2);clf;hold on;
surf(x1_viz, x2_viz, slack_viz, 'FaceColor', 'interp', 'EdgeColor', 'none')
grid on
contour3(x1_viz, x2_viz, slack_viz,...
  25, 'Color', 'k', 'LineWidth', 1);
plot3(x3(max_idx), x4(max_idx), slack(max_idx), 'r.', 'MarkerSize', 30);
xlabel('x3')
ylabel('x4')
h = get(gca,'DataAspectRatio');
set(gca,'DataAspectRatio',[1 1 1/h(1)]);

%% one best configuration

clc;
x_star = make_config(x_task, x3(max_idx), x4(max_idx));

figure(3);clf;
[s, routes, status] = rroptimization(x_star, qos, constrain_slack);
rrsocpinfo(x_star, qos, routes, s);

%% debug configs

% verbosity = 3;
% 
% clc;
% disp('somehow not best solution');
% test_config(qos, rroptimization, x_task, 13/3, 2*13/3, 3);
% disp('best solution');
% test_config(qos, rroptimization, x_task, x1(max_idx), x2(max_idx), 3);

%% helper functions

function x = make_config(x_task, x1, x2)

x = [x_task; x1; 0; x2; 0]; % ensure they're never coincident

end

function test_config(qos, optfunc, x_task, x1, x2, verbosity)

% new team config
x_new = make_config(x_task, x1, x2);

% find unconstrained slack
[slack, routes, ~] = optfunc(x_new, qos, true);

figure(4);
rrsocpinfo(x_new, qos, routes, slack, verbosity);

end
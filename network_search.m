%% parameters
clc;clear;

rrtype = 'meanvar';           % options: meanvar, confidence
constrain_slack = true;       % enforce slack >= 0 during optimization
config_visualization = false; % draw the configs as they are checked
sample_count = 10;            % discretization degree
dist = 10;                    % distance between task agents
x_task = [0 0 dist 0]';       % task team locations
x_comm = [1 0 -1 0]';         % network team locations

if strcmp(rrtype,'meanvar')
  disp('using mean/var formulation');
  rroptimization = @rrsocpmeanvar;

  % communication requirements, agent: 2 -> 1
  qos(1) = struct('flow', struct('src', 2, 'dest', 1),...
    'margin', 0.2,...   % rate margin min
    'confidence', 0.1); % variance bound
else
  disp('using probabilistic confidence formulation');
  rroptimization = @rrsocpconfidence;

  % communication requirements, agent: 2 -> 1
  qos(1) = struct('flow', struct('src', 2, 'dest', 1),...
    'margin', 0.2,...   % rate margin min
    'confidence', 0.7); % probabilistic confidence
end

% entire team config
x = [x_task; x_comm];

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

% fixed x,y translation
T = [dist/2 0]';
T = repmat(T, comm_agent_count, 1);

% form angle and stretch parameter search space
theta = linspace(0,pi,sample_count);
theta = theta - theta(ceil(length(theta)/2)); % subtract the middle element to put 0 in the center
[theta, stretch] = meshgrid(theta, linspace(0.1, dist/2-0.1, sample_count));
theta = theta(:);
stretch = stretch(:);
slack = zeros(size(stretch)); % slack of resulting network

if config_visualization
  figure(1);clf; hold on;
  axis equal;
end

h = waitbar(0, 'Performing parameter search');
for i = 1:length(stretch)
    
  % new team config
  x_new = make_config(x, x_comm, T, Ic, theta(i), stretch(i));
  
  % find unconstrained slack
  [slack(i), ~, ~] = rroptimization(x_new, qos, constrain_slack);
  
  if config_visualization
    plot(x_new(Itx), x_new(Ity), 'r.', 'MarkerSize', 30);
    plot(x_new(Icx), x_new(Icy), 'b.', 'MarkerSize', 30);
    drawnow
  end
  
  waitbar(i/length(stretch),h);
end
close(h)

% a best configuration
[~, max_idx] = max(slack);

%% slack surface visualization

theta_viz = reshape(theta, sample_count*[1 1])*180/pi;
stretch_viz = reshape(stretch, sample_count*[1 1]);
slack_viz = reshape(slack, sample_count*[1 1]);

figure(2);clf;hold on;
surf(theta_viz, stretch_viz, slack_viz)
plot3(theta(max_idx)*180/pi, stretch(max_idx), slack(max_idx), 'r.', 'MarkerSize', 30);
xlabel('theta (deg)')
ylabel('stretch')

%% one best configuration

clc;
x_star = make_config(x, x_comm, T, Ic, theta(max_idx), stretch(max_idx));

figure(3);clf;
[slack, routes, status] = rroptimization(x_star, qos, constrain_slack);
rrsocpinfo(x_star,qos,routes,slack);

%% debug configs

% test_config(qos, rroptimization, x, x_comm, T, Ic, -25.71*pi/180, 3.627);

%% helper functions

function x = make_config(x, x_comm, T, Ic, angle, skew)

% rotation
R = [cos(angle) sin(angle)
    -sin(angle) cos(angle)];
R = kron(eye(length(Ic)/2), R);

% stretching along x direction
D = [skew 0;0 1];
D = kron(eye(length(Ic)/2), D);

% new team config
x(Ic) = R*D*x_comm + T;

end

function test_config(qos, optfunc, x, x_comm, T, Ic, angle, skew)

% new team config
x_new = make_config(x, x_comm, T, Ic, angle, skew);

% find unconstrained slack
[slack, routes, ~] = optfunc(x_new, qos, true);

figure(4);
rrsocpinfo(x_new, qos, routes, slack);

end
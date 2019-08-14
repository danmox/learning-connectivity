%% initialize parameters
clc;clear;

% communication requirements, agent: 2 -> 1
qos(1) = struct('flow', struct('src', 2, 'dest', 1),...
                'margin', 0.2,...
                'confidence', 0.7);

% discretize search
sample_count = 10;

% fixed task agent locations
dist = 8;
x_task = [0 0 dist 0]';

% indexing
task_agent_count = 2;
comm_agent_count = 2;
It = 1:2*task_agent_count; % task_agent_count [x;y] pairs stacked
Itx = It(1:2:end);
Ity = It(2:2:end);
Ic = (1:2*comm_agent_count) + 2*task_agent_count; % comm_agent_count [x;y] pairs stacked
Icx = Ic(1:2:end);
Icy = Ic(2:2:end);

% distribute comm agents about circle (initial config)
angs = (0:2*pi/comm_agent_count:2*pi-pi/comm_agent_count)';
x_comm = reshape([cos(angs) sin(angs)]', [2*comm_agent_count 1]);

% team config
x = zeros(2*(task_agent_count + comm_agent_count), 1);
x(It) = x_task;
x(Ic) = x_comm;

%% parameter search

% fixed x,y translation
T = [dist/2 0]';
T = repmat(T, comm_agent_count, 1);

% form angle and stretch parameter search space
theta = linspace(0,pi,sample_count);
theta = theta - theta(ceil(length(theta)/2)); % subtract the middle element to put 0 in the center
[theta, stretch] = meshgrid(theta,... 
                            linspace(0.1, dist/2-0.1, sample_count));
theta = theta(:);
stretch = stretch(:);
slack = zeros(size(stretch)); % slack of resulting network
status = zeros(size(stretch)); % whether or not the optimization succeeded

h = waitbar(0, 'Performing parameter search');
figure(1);clf; hold on;
axis equal;
for i = 1:length(stretch)
    
  % new team config
  x = make_config(x, x_comm, T, Ic, theta(i), stretch(i));
  
  % find unconstrained slack
  [slack(i), routes, status(i)] = robustroutingsocp(x, qos, false);
   
  plot(x(Itx), x(Ity), 'r.', 'MarkerSize', 30);
  plot(x(Icx), x(Icy), 'b.', 'MarkerSize', 30);
  drawnow
  
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

x_star = make_config(x, x_comm, T, Ic, theta(max_idx), stretch(max_idx));

figure(3);clf;hold on;
plot(x_star(Itx), x_star(Ity), 'r.', 'MarkerSize', 30);
plot(x_star(Icx), x_star(Icy), 'b.', 'MarkerSize', 30);
axis equal

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

% function plot_routes(x,routes)
% 
% figure;hold on;
% plot(x(1:2:numel(x)), x(2:2:numel(x)), 'r.', 'MarkerSize', 30);
% for i = 1:size(x,1)/2
%   for j = i+1:size(x,1)/2
%     xi = x(2*i-1:2*i);
%     xj = x(2*j-1:2*j);
%     plot([xi(1) xj(1)], [xi(2) xj(2)], 'Color', routes(i,j)*ones(3,1),...
%                                        'LineWidth', 2);
%   end
% end
% axis equal
% 
% end
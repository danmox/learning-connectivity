% this function shows the gradient of slack with respect to the
% configuration of the team in order to find locally optimal network
% configurations for fixed task team configurations

%% parameters

clc;clear;

% reward surface / qos / etc.
load('results/100x100_4agents_meanvar_constrainedslack_line_restrictedRx.mat')

sample_count = 100;          % number of local perturbations to consider
x_comm = [[8;0], [12;0]];    % initial network agent locations

% initial team config
x = [x_task x_comm];
x0 = x;

% additional indexing
It = 1:task_agent_count;
Ic = (1:comm_agent_count) + task_agent_count;

N = comm_agent_count + task_agent_count;
K = length(qos);

%% solution at initial configuration

% solve constrained-slack mean/var robust routing formulation
[slack0, routes, ~] = rrsocpmeanvar(x(:), qos, true);

% analysis
figure(2);clf;
rrsocpinfo(x(:), qos, routes, slack0, [0 0 1 1 0])

figure(2);clf;hold on;
surf(x3_viz, x4_viz, slack_viz, 'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', 0.8)
grid on
contour3(x3_viz, x4_viz, slack_viz, 40, 'Color', 'k', 'LineWidth', 1);
plot3(x0(1,Ic(1)), x0(1,Ic(2)), slack(max_idx)+0.1, 'r.', 'MarkerSize', 30);
xlabel('$x_3$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$x_4$', 'Interpreter', 'latex', 'FontSize', 18)
h = get(gca,'DataAspectRatio');
set(gca,'DataAspectRatio', [1 1 1/h(1)]);
drawnow;

%% stochastic hill climbing

std = 0.3;
slack_init = slack0;
slack_star = slack0;
attempts = 0;
total_attempts = 0;
max_attempts = 150;
fprintf('starting slack = %.4f\n', slack_star);
path = zeros(2,0);
samples = zeros(2,0);
while attempts < 25 && total_attempts < max_attempts
    
  x3 = x(1,3) + std*randn;
  x4 = x(1,4) + std*randn;
  
  % make perturbed configuration
  x_pert = make_config(x_task, x3, x4);
  
  % solve SOCP
  slack0 = rrsocpmeanvar(x_pert(:), qos, true);
  
  if slack0 > slack_star
    fprintf('%d samples before improvement\n', attempts+1);
    slack_star = slack0;
    attempts = 0;
    x(1,Ic) = [x3 x4];
    path = [path, [x3;x4]]; %#ok<AGROW>
  else
    samples = [samples, [x_pert(1,3);x_pert(1,4)]]; %#ok<AGROW>
  end
  
  attempts = attempts + 1;
  total_attempts = total_attempts + 1;
  
end
fprintf('total SOCP calls = %d', total_attempts);
fprintf('\nperturbed slack = %.4f\n', slack_star);
fprintf('slack improvement = %.4f\n', slack_star - slack_init);

%% figure

figure(2);clf;hold on;
surf(x3_viz, x4_viz, slack_viz, 'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', 0.8)
grid on
contour3(x3_viz, x4_viz, slack_viz, 40, 'Color', 'k', 'LineWidth', 1);
plot3(samples(1,:), samples(2,:), slack(max_idx)*ones(1,size(samples,2)),...
  '.', 'Color', 0.4*ones(1,3), 'MarkerSize', 10);
plot3([x0(1,3) path(1,:)], [x0(1,4) path(2,:)],...
  slack(max_idx)*ones(1,size(path,2)+1)+0.1, 'r', 'LineWidth', 2);
plot3(x0(1,Ic(1)), x0(1,Ic(2)), slack(max_idx)+0.1, 'r.', 'MarkerSize', 30);
plot3(x(1,Ic(1)),  x(1,Ic(2)),  slack(max_idx)+0.1, 'x', 'MarkerSize', 15,...
  'color', 'r', 'LineWidth', 3);
xlabel('$x_3$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$x_4$', 'Interpreter', 'latex', 'FontSize', 18)
h = get(gca,'DataAspectRatio');
set(gca,'DataAspectRatio', [1 1 1/h(1)]);

%% helper functions

function x = make_config(x_task, x3, x4)

x = [x_task, [x3; 0], [x4; 0]];

end
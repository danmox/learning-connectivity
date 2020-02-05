% this function shows the gradient of slack with respect to the
% configuration of the team in order to find locally optimal network
% configurations for fixed task team configurations

%% parameters

clc;clear;

% reward surface / qos / etc.
load('results/51x51_3agents_probconf_constrainedslack_plane_restrictedRx.mat')

sample_count = 100;    % number of local perturbations to consider
sample_variance = 1.0;
% x_comm = [12;-4];      % initial network agent locations
x_comm = [10;4];      % initial network agent locations

% initial team config
x = [x_task x_comm];
x0 = x;

% additional indexing
It = 1:task_agent_count;
Ic = (1:comm_agent_count) + task_agent_count;

N = comm_agent_count + task_agent_count;
K = length(qos);

%% solution at initial configuration

% solve constrained-slack probabilistic confidence robust routing formulation
[slack0, routes, ~] = rrsocpprobconf(x(:), qos, true);

% analysis
figure(2);clf;
rrsocpinfo(x(:), qos, routes, slack0, [0 0 1 1 0])

figure(2);clf;hold on;
surf(x_viz, y_viz, slack_viz, 'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', 0.8)
grid on
contour3(x_viz, y_viz, slack_viz, 40, 'Color', 'k', 'LineWidth', 1);
plot3(x0(1,Ic), x0(2,Ic), slack(max_idx), 'r.', 'MarkerSize', 30);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 18)
h = get(gca,'DataAspectRatio');
set(gca,'DataAspectRatio', [1 1 1/h(1)]);
drawnow;

%% v surface visualization

% xspace = linspace(0, dist, sample_count);
% yspace = xspace - xspace(ceil(length(xspace)/2));
% [xs, ys] = meshgrid(xspace, yspace);
% xs = xs(:);
% ys = ys(:);
% vs = zeros(size(ys));
% for i = 1:length(xs)
%   xnew = make_config(x_task, [xs(i); ys(i)]);
%   vs(i) = computeV(qos, routes, xnew);
% end
% x_viz = reshape(xs, length(xspace)*ones(1,2));
% y_viz = reshape(ys, length(xspace)*ones(1,2));
% v_viz = reshape(vs, length(xspace)*ones(1,2));
% 
% figure(3);clf;hold on;
% surf(x_viz, y_viz, v_viz,...
%   'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', 0.8)
% grid on
% contour3(x_viz, y_viz, v_viz, 40, 'Color', 'k', 'LineWidth', 1);
% plot3(x0(1,Ic), x0(2,Ic), max(vs(:))+0.1, 'r.', 'MarkerSize', 30);
% xlabel('$x_3$', 'Interpreter', 'latex', 'FontSize', 18)
% ylabel('$x_4$', 'Interpreter', 'latex', 'FontSize', 18)
% h = get(gca,'DataAspectRatio');
% set(gca,'DataAspectRatio', [1 1 1/h(1)]);
% drawnow;

%% find locally optimal configurations w.r.t the current routing solution
% [RUN THIS SECTION REPEATEDLY (CTRL+ENTER) TO PLAN
clc

x_star = zeros(size(x));
bik_star = computebik(qos, routes, x);
for i = 1:sample_count
  
  x3 = x(:,Ic) + sample_variance*randn(2,comm_agent_count);
  
  % make perturbed configuration
  x_pert = make_config(x_task, x3);
  
  bik = computebik(qos, routes, x_pert);
  
  if bik > bik_star
    x_star = x_pert;
    bik_star = bik;
  end
  
end
if ~all(x_star(:) == 0)
  disp(x_star);
  x = x_star;
end

% new solution
[slack0, routes, ~] = rrsocpprobconf(x(:), qos, true);
rrsocpinfo(x(:), qos, routes, slack0, [0 0 1 1 0])

% figure

% figure(2);clf;hold on;
% colormap('parula')
% surf(x_viz, y_viz, slack_viz, 'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', 0.8)
% colorbar
% grid on
% contour3(x_viz, y_viz, slack_viz, 40, 'Color', 'k') %, 'LineWidth', 1);
plot3(x(1,Ic), x(2,Ic), slack(max_idx)+0.1, 'r.', 'MarkerSize', 30);
% xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 20)
% ylabel('$xy$', 'Interpreter', 'latex', 'FontSize', 20)
% h = get(gca,'DataAspectRatio');
% set(gca,'DataAspectRatio', [1 1 1/h(1)]);

%% helper functions

function x = make_config(x_task, x3)

x = [x_task, x3];

end

function b_ik_min = computebik(qos, routes, x)

N = size(routes,1);
K = length(qos);

R = linkratematrix(x(:));
[~,B] = nodemarginconsts(qos,R);

% useful for constructing constraints (since destination nodes have no
% constraints)
dest_mask = false(N,K);
for k = 1:K
  dest_mask(qos(k).flow.dest,k) = true;
end
dest_mask = reshape(dest_mask, [N*K 1]);

% confidence threshold
conf = ones(N,K).*norminv(horzcat(qos(:).confidence), 0, 1);
conf = reshape(conf, [N*K 1]);
conf(dest_mask) = [];

% node margins
m_ik = zeros(N,K);
for k = 1:K
  m_ik(qos(k).flow.src,k) = qos(k).margin;
end
m_ik = reshape(m_ik, [N*K 1]);
m_ik(dest_mask) = [];

y = [full(routes(:)); 0];
b_ik_min = min(B*y - m_ik);

end

function v = computeV(qos, routes, x)

N = size(routes,1);
K = length(qos);

R = linkratematrix(x(:));
[A,B,~] = nodemarginconsts(qos,R);

% useful for constructing constraints (since destination nodes have no
% constraints)
dest_mask = false(N,K);
for k = 1:K
  dest_mask(qos(k).flow.dest,k) = true;
end
dest_mask = reshape(dest_mask, [N*K 1]);

% confidence threshold
conf = ones(N,K).*norminv(horzcat(qos(:).confidence), 0, 1);
conf = reshape(conf, [N*K 1]);
conf(dest_mask) = [];

% node margins
m_ik = zeros(N,K);
for k = 1:K
  m_ik(qos(k).flow.src,k) = qos(k).margin;
end
m_ik = reshape(m_ik, [N*K 1]);
m_ik(dest_mask) = [];

y = [full(routes(:)); 0];
v_ik = (B*y - m_ik)./vecnorm(A.*(y'),2,2) - conf;
v = min(v_ik);

end
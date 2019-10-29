% this script performs gradient ascent in order to find locally optimal
% network configurations given a fixed task team configuration

%% parameters

clc;clear;

% reward surface / qos / etc.
load('results/50x50_3agents_meanvar_constrainedslack_plane_restrictedRx')

x = make_config(x_task, [10;4]);
x0 = x;

%% solution at initial configuration

% solve constrained-slack mean/var robust routing formulation
[slack0, routes, ~] = rrsocpmeanvar(x(:), qos, true);

% analysis
figure(2);clf;
rrsocpinfo(x(:), qos, routes, slack0, [1 0 1 1 0])

figure(2);clf;hold on;
surf(x_viz, y_viz, slack_viz, 'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceAlpha', 0.8)
grid on
contour3(x_viz, y_viz, slack_viz, 40, 'Color', 'k', 'LineWidth', 1);
plot3(x0(1,Ic(1)), x0(2,Ic(1)), slack(max_idx)+0.1, 'r.'); %, 'MarkerSize', 30);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 18)
h = get(gca,'DataAspectRatio');
set(gca,'DataAspectRatio', [1 1 1/h(1)]);
drawnow;

%% gradient-ascent

if issparse(routes)
  routes = full(routes);
end

N = comm_agent_count + task_agent_count;
K = length(qos);

% node margin bounds
m_ik = zeros(N,K);
for k = 1:K
  m_ik(union(qos(k).flow.src, qos(k).flow.dest),k) = qos(k).margin;
end

% constraint matrices (to determine which are active)
[A,B,zero_vars] = nodemarginconsts(qos, linkratematrix(x(:)));
y = [routes(:); 0];

% compute gradient
mean_grad = zeros(2,N);
var_grad = zeros(2,N);
for k = 1:K
  for i = 1:N
    
    % determine if any constraints are active
    n = (k-1)*N + i;
    mean_lhs = B(n,:)*y - m_ik(i,k);
    var_lhs = norm(diag(A(n,:))*y);
    
    % active bandwidth constraint
    if abs(mean_lhs - slack0) < 1e-6
      fprintf('mean constraint %d is active\n', i);     
      for n = Ic
        mean_grad(:,n) = mean_grad(:,n) + margin_exp_derivative(x, routes, B(n,:), i, k, n);
      end
    end
    
    % active variance constraint
    if abs(var_lhs - sqrt(qos(k).confidence)) < 1e-6
      fprintf('var constraint %d is active\n', i);
      for n = Ic
        var_grad(:,n) = var_grad(:,n) + margin_var_derivative(x, routes, i, k, n);
      end
    end
    
  end
end
grad = mean_grad - var_grad; % increasing mean, decreasing variance

% visualize
quiver3(x(1,Ic(1)), x(2,Ic(1)), slack(max_idx), grad(1,3), grad(2,3), 0,...
  4,...
  'LineWidth', 1.5, 'Color', 'r')

% take step
grad_step = 0.1;
x = x + grad_step*grad; % only network agents have non-zero gradient
plot3(x(1,Ic(1)), x(2,Ic(1)), slack(max_idx), 'r.', 'MarkerSize', 10);

% find new routing variables
[slack0, routes, ~] = rrsocpmeanvar(x(:), qos, true);
rrsocpinfo(x(:), qos, routes, slack0, [0 0 1 1 0])

%% helper functions

function dexp = margin_exp_derivative(x, routes, Bik, i, k, n)
% derivative of margin mean i for flow k with respect to agent n

dexp = zeros(2,1);

sgn = sign(reshape(Bik(1:end-1), size(routes(:,:,1))));

for j = 1:size(x,2)

  if i == n
    dexp = dexp + sgn(i,j) * routes(i,j,k) * channel_derivative(x(:,i), x(:,j), 1, 'mean'); % outgoing
    dexp = dexp + sgn(j,i) * routes(j,i,k) * channel_derivative(x(:,j), x(:,i), 2, 'mean'); % incoming
  elseif j == n
    dexp = dexp + sgn(i,j) * routes(i,j,k) * channel_derivative(x(:,i), x(:,j), 2, 'mean'); % outgoing
    dexp = dexp + sgn(j,i) * routes(j,i,k) * channel_derivative(x(:,j), x(:,i), 1, 'mean'); % incoming
  end

end

end

function dvar = margin_var_derivative(x, routes, i, k, n)
% derivative of margin variance i for flow k with respect to agent n

dvar = zeros(2,1);

for j = 1:size(x,2)
  
  if i == n
    dvar = dvar + routes(i,j,k)^2 * channel_derivative(x(:,i), x(:,j), 1, 'var'); % outgoing
    dvar = dvar + routes(j,i,k)^2 * channel_derivative(x(:,j), x(:,i), 2, 'var'); % incoming
  elseif j == n
    dvar = dvar + routes(i,j,k)^2 * channel_derivative(x(:,i), x(:,j), 2, 'var'); % outgoing
    dvar = dvar + routes(j,i,k)^2 * channel_derivative(x(:,j), x(:,i), 1, 'var'); % incoming
  end
  
end

end

function x = make_config(x_task, x3)

x = [x_task, x3];

end
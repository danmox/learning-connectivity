% this function shows the gradient of slack with respect to the
% configuration of the team in order to find locally optimal network
% configurations for fixed task team configurations

%% parameters

clc;clear;

% reward surface / qos / etc.
load('results/100x100_4agents_meanvar_constrainedslack_line_restrictedRx.mat')

% initial network team configuration

% initial team config
x = [x_task x_comm];

%% solution at initial configuration

% solve constrained-slack mean/var robust routing formulation
[slack0, routes, ~] = rrsocpmeanvar(x(:), qos, true);

% analysis
figure(2);clf;
rrsocpinfo(x(:), qos, routes, slack0, [0 0 1 1 0])

% plotting
figure(2);clf;hold on;
surf(x3_viz, x4_viz, slack_viz, 'FaceColor', 'interp', 'EdgeColor', 'none')
contour3(x3_viz, x4_viz, slack_viz, 25, 'Color', 'k', 'LineWidth', 1);
plot3(x(1,Ic(1)), x(1,Ic(2)), slack(max_idx), 'r.', 'MarkerSize', 20);
h = get(gca,'DataAspectRatio');
set(gca,'DataAspectRatio',[1 1 1/h(1)*2]);
xlabel('$x_{3,x}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$x_{4,x}$', 'Interpreter', 'latex', 'FontSize', 16);

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
[A,B] = nodemarginconsts(qos, linkratematrix(x(:)));
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
        mean_grad(:,n) = mean_grad(:,n) + margin_derivative(x, routes, i, k, n);
      end
    end
    
    % active variance constraint
    if abs(var_lhs - sqrt(qos(k).confidence)) < 1e-6
      fprintf('var constraint %d is active\n', i);
      for n = Ic
        var_grad(:,n) = var_grad(:,n) + variance_derivative(x, routes, i, k, n);
      end
    end
    
  end
end
grad = mean_grad - var_grad; % increasing mean, decreasing variance

% visualize
quiver3(x(1,Ic(1)), x(1,Ic(2)), slack(max_idx),...
  grad(1,3), grad(1,4), 0,...
  'LineWidth', 1.5, 'Color', 'r')

% take step
x = x + grad_step*grad; % only network agents have non-zero gradient
plot3(x(1,Ic(1)), x(1,Ic(2)), slack(max_idx), 'r.', 'MarkerSize', 10);

% find new routing variables
[slack0, routes, ~] = rrsocpmeanvar(x(:), qos, true);
rrsocpinfo(x(:), qos, routes, slack0, [0 0 1 1 0])

%% helper functions

function dbik_dxn = margin_derivative(x, routes, i, k, n)
% derivative of margin mean i for flow k with respect to agent n

dbik_dxn = zeros(2,1);

for j = 1:size(x,2)
  
  if i == n
    dbik_dxn = dbik_dxn + routes(i,j,k) * channel_derivative(x(:,i), x(:,j), 1, 'mean'); % outgoing
    dbik_dxn = dbik_dxn - routes(j,i,k) * channel_derivative(x(:,j), x(:,i), 2, 'mean'); % incoming
  elseif j == n
    dbik_dxn = dbik_dxn + routes(i,j,k) * channel_derivative(x(:,i), x(:,j), 2, 'mean'); % outgoing
    dbik_dxn = dbik_dxn - routes(j,i,k) * channel_derivative(x(:,j), x(:,i), 1, 'mean'); % incoming
  end
    
end

end

function dbik_dxn = variance_derivative(x, routes, i, k, n)
% derivative of margin variance i for flow k with respect to agent n

dbik_dxn = zeros(2,1);

for j = 1:size(x,2)
  
  if i == n
    dbik_dxn = dbik_dxn + routes(i,j,k)^2 * channel_derivative(x(:,i), x(:,j), 1, 'var'); % outgoing
    dbik_dxn = dbik_dxn + routes(j,i,k)^2 * channel_derivative(x(:,j), x(:,i), 2, 'var'); % incoming
  elseif j == n
    dbik_dxn = dbik_dxn + routes(i,j,k)^2 * channel_derivative(x(:,i), x(:,j), 2, 'var'); % outgoing
    dbik_dxn = dbik_dxn + routes(j,i,k)^2 * channel_derivative(x(:,j), x(:,i), 1, 'var'); % incoming
  end
  
end

end
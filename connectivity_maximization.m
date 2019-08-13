%% initialize parameters

task_agent_count = 3;
network_agent_count = 2;

N = task_agent_count + network_agent_count;
It = 1:task_agent_count; % set of task node indices
In = (1:network_agent_count) + task_agent_count; % set of network node indices

% nodes = zeros(N,2);
% for i = It
%   nodes(i,1) = cos((i-1)*2*pi/task_agent_count);
%   nodes(i,2) = sin((i-1)*2*pi/task_agent_count);
% end
% nodes = nodes + 1;
nodes = zeros(N,2);
nodes(It,:) = rand(task_agent_count,2)-0.5;

X = nodes';
G = X'*X;
Dsqr = diag(G)*ones(N,1)' + ones(N,1)*diag(G)' - 2*G;
d_max = max(Dsqr(:));
L = diag(Dsqr*ones(N,1)) - Dsqr;

%% visualization

figure(1);clf;hold on;
for i = 1:size(nodes,1)
    for j = i+1:size(nodes,1)
        plot(nodes([i j],1), nodes([i j],2), '-', 'Color', 0.8*ones(1,3));
    end
end
plot(nodes(It,1), nodes(It,2), '.', 'Color', [1.0 0.0 0.0], 'MarkerSize', 30)
plot(nodes(In,1), nodes(In,2), '.', 'Color', [0.7 0.7 1.0], 'MarkerSize', 30)
axis equal

%% optimization problem

V = eye(N) - 1/N*ones(N);
var_mask = false(N) | logical(eye(N));
var_mask(It,It) = true;
cvx_begin quiet
    variable EDM(N,N) symmetric nonnegative
    variable u nonnegative
    expression A(N,N)
    A = ones(N)-eye(N)-1/d_max*EDM;
    maximize(u)
    subject to
        -V*EDM*V == semidefinite(N)
        diag(A*ones(N,1)) - A - u*V == semidefinite(N)
        EDM(var_mask) == Dsqr(var_mask)
cvx_end

%% list reconstruction

Xstar = [nodes(It,:); zeros(network_agent_count, 2)];
for i = In
  A = 2*nodes(It,:);
  b = sum(nodes(It,:).^2,2) - EDM(It,i);
  D = [-ones(task_agent_count-1,1) eye(task_agent_count-1)];
  Xstar(i,:) = ((D*A)'*D*A)\((D*A)'*D*b);
end


%% final visualization

figure(2);clf;hold on;
for i = 1:size(Xstar,1)
    for j = i+1:size(Xstar,1)
        plot(Xstar([i j],1), Xstar([i j],2), 'Color', [0.8 0.8 0.8]);
    end
end
plot(Xstar(It,1), Xstar(It,2), '.', 'Color', [1.0 0.0 0.0], 'MarkerSize', 30)
plot(Xstar(In,1), Xstar(In,2), '.', 'Color', [0.0 0.0 1.0], 'MarkerSize', 30)
axis equal
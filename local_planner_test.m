%% parameters
clc;clear;

dist = 15;
x = [[0;0], [dist;0], [dist/2;0]];

% communication requirements, agent: 2 -> 1
qos(1) = struct('flow', struct('src', 2, 'dest', 1),...
  'margin', 0.25,...
  'confidence', 0.90);
qos(2) = struct('flow', struct('src', 1, 'dest', 2),...
  'margin', 0.25,...
  'confidence', 0.90);

N = size(x,2);
K = length(qos);

% confidence threshold
conf = ones(N,K).*norminv(horzcat(qos(:).confidence), 0, 1);
conf = reshape(conf, [N*K 1]);

% node margins
m_ik = zeros(N,K);
for k = 1:K
  m_ik([qos(k).flow.src qos(k).flow.dest],k) = qos(k).margin;
end
m_ik = reshape(m_ik, [N*K 1]);

%% slack test

x(:,3) = [dist/2; 4.0];
[slack, routes, ~] = rrsocpprobconf(x(:), qos, true);
rrsocpinfo(x(:), qos, routes, slack, [0 0 1 0 1])

slack = slack

x(:,3) = [dist/2; 2.0];
est_slack = min(computeSlack(qos, x, m_ik, conf, routes))
true_slack = rrsocpprobconf(x(:), qos, true)

%% helper functions

function s = computeSlack(qos, x, m_ik, conf, routes)

% active constraints
R = linkratematrix(x(:));
[A,B,~] = nodemarginconsts(qos,R);

num_const = size(A,1);
s = zeros(num_const,1);
y = [routes(:); 0];
for j = 1:num_const
  s(j) = B(j,:)*y - m_ik(j) - conf(j)*norm(diag(A(j,:))*y);
end

end

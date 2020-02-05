function runlocalcontroller(x_star, qos, Ic, sample_count, sample_variance, its)

N = size(x_star, 2);
K = length(qos);
comm_agent_count = length(Ic);
It = setdiff(1:N, Ic);

% confidence threshold
conf = ones(N,K).*norminv(horzcat(qos(:).confidence), 0, 1);
conf = reshape(conf, [N*K 1]);

% node margins
m_ik = zeros(N,K);
for k = 1:K
  m_ik([qos(k).flow.src qos(k).flow.dest],k) = qos(k).margin;
end
m_ik = reshape(m_ik, [N*K 1]);

% initialize slack figure
figure(1); clf; hold on;
slack_ax = gca;

% config figure bounds
axis_bounds = [min(x_star(1,:)) max(x_star(1,:)) min(x_star(2,:)) max(x_star(2,:))];

for it = 1:its

  fprintf('\niteration %d\n', it);

  % find optimal routing variables for current position

  [slack, routes, ~] = rrsocpprobconf(x_star(:), qos, true);
  fprintf('slack = %.4f\n', slack);
  routes = full(routes);
  if it == 1
    plot(slack_ax, it, slack, 'bx', 'MarkerSize', 10, 'LineWidth', 2);
  else
    plot(slack_ax, [-1 0]+it, [slack0 slack], 'b', 'LineWidth', 2);
  end
  drawnow
  slack0 = slack;
  slack_star = slack;

  % update network configuration

  x0 = x_star;
  sample_hist = zeros(2, comm_agent_count, sample_count);
  slack_updates = 0;
  for i = 1:sample_count

    % perturb configuration

    x_prime = x0;
    x_prime(:, Ic) = x_prime(:, Ic) + sample_variance*randn(2, comm_agent_count);
    sample_hist(:, :, i) = x_prime(:, Ic);

    % compute slack

    slack_prime = min(computeSlack(qos, x_prime, m_ik, conf, routes));

    % update best config

    if slack_prime - slack_star > 0.001
      slack_updates = slack_updates + 1;
      slack_star = slack_prime;
      x_star(:, Ic) = x_prime(:, Ic);
    end

  end
  fprintf('slack_star updated %d times\n', slack_updates);

  % planning visualization

  figure(2); clf; hold on;
  for i = 1:comm_agent_count
    plot(reshape(sample_hist(1,i,:), [sample_count 1]),...
      reshape(sample_hist(2,i,:), [sample_count 1]), '.', 'Color', [1 0.8 0.8]);
  end
  plot(x_star(1, It), x_star(2, It), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
  plot(x0(1, Ic), x0(2, Ic), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
  plot(x_star(1, Ic), x_star(2, Ic), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
  axis equal
  axis(axis_bounds)
  drawnow

%   % analysis
%   [slack_star, routes_star] = rrsocpprobconf(x_star(:), qos, true);
%   if slack_star < slack
%
%     s0 = computeSlack(qos, x0, m_ik, conf, routes);
%     s = computeSlack(qos, x_star, m_ik, conf, routes);
%     s_star = computeSlack(qos, x_star, m_ik, conf, routes_star);
%     fprintf('slack0 = %.4f, slack_star = %.4f\n', slack, slack_star);
%     table(round(s0,4), round(s,4), round(s_star,4), 'VariableNames', {'s0','s','s_star'})
%   end

end

end

% helper functions

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

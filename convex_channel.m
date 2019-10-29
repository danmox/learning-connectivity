% this script compares different approximate, convex channel models with
% the non-convex approximate stochastic channel model used by Fink

x = 0:0.1:40; % distance between agents

% build stochastic confidence regions
Uij_approx = zeros(5,length(x));
[Rij, Vij] = channel_model(x);
counter = 1;
for k = -2:2
  Uij_approx(counter,:) = Rij + k*sqrt(Vij);
  counter = counter + 1;
end
Uij_approx(:,1) = 1;

% plot stochastic model and confidence regions
figure(1); clf; hold on;

fill([x, fliplr(x)], [Uij_approx(1,:), fliplr(Uij_approx(5,:))], 'c', ...
      'EdgeColor', 'none', 'FaceColor', 0.8*[1 1 1]);
fill([x, fliplr(x)], [Uij_approx(2,:), fliplr(Uij_approx(4,:))], 'm', ...
      'EdgeColor', 'none', 'FaceColor', 0.6*[1 1 1]);
    
plot(x, Uij_approx(3, :), 'k', 'LineWidth', 2, 'DisplayName', 'Stochastic');
axis([0 30 -0.2 1.5])
ylabel('R_{ij}', 'FontSize', 16)
xlabel('d (m)', 'FontSize', 16)

plot(x,linear_channel(x),'LineWidth', 2, 'DisplayName', 'linear');
plot(x,sigmoid_channel(x),'LineWidth', 2, 'DisplayName', 'sigmoid');


function R = linear_channel(d)

R = 1-d/25;
R(R < 0) = 0.0;

end

function R = sigmoid_channel(d)

R = 2 - 2 ./ (1 + exp(-d/7));

end

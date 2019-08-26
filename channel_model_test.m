% this script compares the true stochastic channel model with its
% approximation

clear;clc;

%% model parameters

cm = channel_model();

%% true channel rate mean and confidence levels

x = 0:0.01:30; % meters
Uij_true = zeros(5,length(x));
counter = 1;
for k = -2:2
  P_R = dBm2mW(cm.L0 - 10*cm.n*log10(x) + k*sqrt(cm.sigma_F2));
  Uij_true(counter,:) = 1 - erfc(sqrt(P_R/cm.PN0));
  counter = counter + 1;
end

%% approximate mean and confidence levels

Uij_approx = zeros(5,length(x));
[Rij, Vij] = stochastic_channel(x);
counter = 1;
for k = -2:2
  Uij_approx(counter,:) = Rij + k*sqrt(Vij);
  counter = counter + 1;
end
Uij_approx(:,1) = 1;

%% plot overlay

figure(1); clf; hold on;
alpha = 0.5;

% true confidence bounds
f1 = fill([x, fliplr(x)], [Uij_true(1,:), fliplr(Uij_true(5,:))], 'c',...
     'EdgeColor', 'none',...
     'FaceColor', 'r',...
     'FaceAlpha', alpha*1/3);
f2 = fill([x, fliplr(x)], [Uij_true(2,:), fliplr(Uij_true(4,:))], 'm',...
     'EdgeColor', 'none',...
     'FaceColor', 'r',...
     'FaceAlpha', alpha*2/3);

% approximate confidence bounds
f3 = fill([x, fliplr(x)], [Uij_approx(1,:), fliplr(Uij_approx(5,:))], 'c',...
     'EdgeColor', 'none',...
     'FaceColor', 'b',...
     'FaceAlpha', alpha*1/3);
f4 = fill([x, fliplr(x)], [Uij_approx(2,:), fliplr(Uij_approx(4,:))], 'm',...
     'EdgeColor', 'none',...
     'FaceColor', 'b',...
     'FaceAlpha', alpha*2/3);

% true/approx mean
p1 = plot(x, Uij_true(3, :), 'r', 'LineWidth', 2);
p2 = plot(x, Uij_approx(3, :), 'b', 'LineWidth', 2);

axis([0 30 -0.2 1.5])
legend([p1 f2 f1 p2 f4 f3], {'true', '\sigma', '2\sigma', 'approx', '\sigma', '2\sigma'}, 'FontSize', 16);
ylabel('R_{ij}', 'FontSize', 16)
xlabel('d (m)', 'FontSize', 16)

%% helper functions

function mW = dBm2mW(dBm)

mW = 10.^(dBm/10);

end
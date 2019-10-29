% this script compares the true stochastic channel model with its
% approximation

clear;clc;

%% model parameters

cm = channel_params();

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
Uij_approx2 = zeros(5,length(x));
[Rij, ~] = channel_model(x);

% compute link rate
PR = cm.L0 - 10*cm.n*log10(x); % dBm
P = dBm2mW(PR); % mW
rate = 1 - erfc(sqrt(P./cm.PN0));

% compute link variance (using delta method approximation)
Vij = (log(10)/(10*sqrt(cm.PN0*pi))*exp(-P/cm.PN0).*10.^(PR/20)).^2*cm.sigma_F2;
Vij(isnan(Vij)) = 0;
Vij2 = (0.2*x./(6 + x)).^2;
counter = 1;
for k = -2:2
  Uij_approx(counter,:) = Rij + k*sqrt(Vij);
  Uij_approx2(counter,:) = Rij + k*sqrt(Vij2);
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

% % approximate confidence bounds
% f3 = fill([x, fliplr(x)], [Uij_approx(1,:), fliplr(Uij_approx(5,:))], 'c',...
%      'EdgeColor', 'none',...
%      'FaceColor', 'g',...
%      'FaceAlpha', alpha*1/3);
% f4 = fill([x, fliplr(x)], [Uij_approx(2,:), fliplr(Uij_approx(4,:))], 'm',...
%      'EdgeColor', 'none',...
%      'FaceColor', 'g',...
%      'FaceAlpha', alpha*2/3);

% approximate confidence bounds 2
f5 = fill([x, fliplr(x)], [Uij_approx2(1,:), fliplr(Uij_approx2(5,:))], 'c',...
     'EdgeColor', 'none',...
     'FaceColor', 'b',...
     'FaceAlpha', alpha*1/3);
f6 = fill([x, fliplr(x)], [Uij_approx2(2,:), fliplr(Uij_approx2(4,:))], 'm',...
     'EdgeColor', 'none',...
     'FaceColor', 'b',...
     'FaceAlpha', alpha*2/3);

% true/approx mean
p1 = plot(x, Uij_true(3, :), 'k', 'LineWidth', 2);
ax = gca;
ax.FontSize = 16;

axis([0 30 -0.2 1.5])
legend([p1 f2 f1 f6 f5],...
  {'mean', 'model $\sqrt{\sigma}$', 'model $2\sqrt{\sigma}$',...
  'sigmoid $\sqrt{\sigma}$', 'sigmoid $2\sqrt{\sigma}$'},...
  'FontSize', 24, 'Interpreter', 'latex');
ylabel('$R_{ij}$', 'FontSize', 24, 'Interpreter', 'latex')
xlabel('$d$ (m)', 'FontSize', 24, 'Interpreter', 'latex')

%% variance plot

figure(2);clf;hold on;
plot(x, sqrt(Vij), 'r', 'LineWidth', 2);
plot(x, sqrt(Vij2), 'b', 'LineWidth', 2);
legend({'delta', 'sigmoid'}, 'Interpreter', 'latex', 'FontSize', 22)
xlabel('$d$ (m)', 'Interpreter', 'latex', 'FontSize', 24)
ylabel('$\sqrt{\tilde{R}_{ij}}$', 'Interpreter', 'latex', 'FontSize', 24)
ax = gca;
ax.FontSize = 16;

%% model plot

figure(3);clf;hold on;

plot(x, Uij_true(3, :), 'r', 'LineWidth', 2);
plot(x, sqrt(Vij2), 'b', 'LineWidth', 2);
legend({'$\bar{R}_{ij}$', '$\sqrt{\tilde{R}_{ij}}$'}, 'Interpreter', 'latex', 'FontSize', 22, 'box', 'off')
xlabel('$d$ (m)', 'Interpreter', 'latex', 'FontSize', 24)
ax = gca;
ax.FontSize = 16;
axis([0 30 0 1.0])
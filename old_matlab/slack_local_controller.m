%% 2 task 1 network line test

clc;clear;

dist = 15;
x = [[0;0], [dist;0], [dist/2;4]];
Ic = 3;
sample_count = 100;
sample_variance = 0;
its = 50;

% communication requirements, agent: 2 -> 1
qos(1) = struct('flow', struct('src', 2, 'dest', 1),...
  'margin', 0.25,...
  'confidence', 0.90);
qos(2) = struct('flow', struct('src', 1, 'dest', 2),...
  'margin', 0.25,...
  'confidence', 0.90);

runlocalcontroller(x, qos, Ic, sample_count, sample_variance, its);
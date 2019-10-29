%% 3 task 1 network patrol

filename = 'bags/3task1network_2019-09-22-20-21-41';

bag = rosbag([filename '.bag']);
qosbag = select(bag, 'Topic', '/qos');
qos = zeros(qosbag.NumMessages, 4);
qos(:,1) = qosbag.MessageList.Time;
qos(:,1) = qos(:,1) - qos(1,1);
msg1 = readMessages(qosbag,1);
margin = msg1{1}.Data(1);
confidence = msg1{1}.Data(3);
for i = 1:qosbag.NumMessages
  msg = readMessages(qosbag,i);
  msg_data = msg{1}.Data';
  qos(i,2) = mean(msg_data(2:4:end)); % margin
  qos(i,3) = mean(sqrt(msg_data(4:4:end))); % std
  qos(i,4) = norminv(confidence,0,1)*qos(i,3); % confidence interval
end

figure(1);clf;hold on
plot(qos(:,1), qos(:,2), 'b', 'LineWidth', 4)
fill([qos(:,1); flipud(qos(:,1))], [qos(:,2) - qos(:,4); flipud(qos(:,2) + qos(:,4))],'c',...
     'EdgeColor', 'none',...
     'FaceColor', 'b',...
     'FaceAlpha', 0.5);
% plot(qos(:,1), qos(:,2) - qos(:,4), 'LineWidth', 4);
plot([0 qos(end,1)], margin*ones(1,2), 'k:', 'LineWidth', 4)
ax = gca;
ax.FontSize = 24;
axis_limits = axis;
axis_limits([3 4]) = [0 0.2];
axis(axis_limits)
xlabel('time (seconds)', 'interpreter', 'latex', 'FontSize', 40)
ylabel('rate margin', 'interpreter', 'latex', 'FontSize', 40)
legend({'rate', 'confidence', 'min rate'}, 'interpreter', 'latex',...
  'location', 'NorthEastOutside', 'FontSize', 30)

%% 3 task 3 network patrol

% fixed network agents

filename = 'bags/3task3network_2019-09-22-20-26-57';

bag = rosbag([filename '.bag']);
qosbag = select(bag, 'Topic', '/qos');
qos331 = zeros(qosbag.NumMessages, 4);
qos331(:,1) = qosbag.MessageList.Time;
qos331(:,1) = qos331(:,1) - qos331(1,1);
msg1 = readMessages(qosbag,1);
confidence = msg1{1}.Data(3);
for i = 1:qosbag.NumMessages
  msg = readMessages(qosbag,i);
  msg_data = msg{1}.Data';
  qos331(i,2) = mean(msg_data(2:4:end)); % margin
  qos331(i,3) = mean(sqrt(msg_data(4:4:end))); % std
  qos331(i,4) = norminv(confidence,0,1)*qos331(i,3); % confidence interval
end

% moving network agents

filename = 'bags/3task3network_2019-09-22-21-00-52';

bag = rosbag([filename '.bag']);
qosbag = select(bag, 'Topic', '/qos');
qos332 = zeros(qosbag.NumMessages, 4);
qos332(:,1) = qosbag.MessageList.Time;
qos332(:,1) = qos332(:,1) - qos332(1,1);
msg1 = readMessages(qosbag,1);
margin = msg1{1}.Data(1);
confidence = msg1{1}.Data(3);
for i = 1:qosbag.NumMessages
  msg = readMessages(qosbag,i);
  msg_data = msg{1}.Data';
  qos332(i,2) = mean(msg_data(2:4:end)); % margin
  qos332(i,3) = mean(sqrt(msg_data(4:4:end))); % std
  qos332(i,4) = norminv(confidence,0,1)*qos332(i,3); % confidence interval
end

% plots

figure(2);clf;hold on
plot(qos331(:,1), qos331(:,2), 'r', 'LineWidth', 2)
plot(qos331(:,1), qos331(:,2) - qos331(:,4), 'r:', 'LineWidth', 2);
plot(qos332(:,1), qos332(:,2), 'b', 'LineWidth', 2)
plot(qos332(:,1), qos332(:,2) - qos332(:,4), 'b:', 'LineWidth', 2);
plot([0 max(qos332(end,1), qos331(end,1))], margin*ones(1,2))
axis_limits = axis;
axis_limits([3 4]) = [0 0.4];
axis(axis_limits)

figure(2);clf;hold on
plot(qos331(:,1), qos331(:,2), 'r', 'LineWidth', 4)
fill([qos331(:,1); flipud(qos331(:,1))],...
     [qos331(:,2) - qos331(:,4); flipud(qos331(:,2) + qos331(:,4))],...
     'c',...
     'EdgeColor', 'none',...
     'FaceColor', 'r',...
     'FaceAlpha', 0.5);
plot(qos332(:,1), qos332(:,2), 'b', 'LineWidth', 4)
fill([qos332(:,1); flipud(qos332(:,1))],...
     [qos332(:,2) - qos332(:,4); flipud(qos332(:,2) + qos332(:,4))],...
     'c',...
     'EdgeColor', 'none',...
     'FaceColor', 'b',...
     'FaceAlpha', 0.5);
plot([0 max(qos331(end,1), qos332(end,1))], margin*ones(1,2), 'k:', 'LineWidth', 4)
ax = gca;
ax.FontSize = 24;
axis_limits = axis;
axis_limits([3 4]) = [0 0.4];
axis(axis_limits)
xlabel('time (seconds)', 'interpreter', 'latex', 'FontSize', 40)
ylabel('rate margin', 'interpreter', 'latex', 'FontSize', 40)
legend({'fixed rate', 'fixed confidence', 'moving rate', 'moving confidence', 'min rate'},...
  'location', 'NorthEastOutside', 'interpreter', 'latex', 'FontSize', 30)


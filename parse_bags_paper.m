%% 3 task agents 1 network agent patrol

margin = 0.025;
confidence = 0.70;
filename = 'bags/3t1n_paper';

panbag = rosbag([filename '.bag']);
qosbag = select(panbag, 'Topic', '/qos');
msg1 = readMessages(qosbag,1);
flows = length(msg1{1}.Data)/2;
qos = zeros(qosbag.NumMessages, flows*3+1);
qos(:,1) = qosbag.MessageList.Time;
qos(:,1) = qos(:,1) - qos(1,1);
for i = 1:qosbag.NumMessages
  msg = readMessages(qosbag,i);
  msg_data = msg{1}.Data';
  qos(i,2:3:end) = msg_data(1:2:end);
  qos(i,3:3:end) = sqrt(msg_data(2:2:end));
  qos(i,4:3:end) = norminv(confidence,0,1)*qos(i,3:3:end);
end

figure(1);clf;hold on
plot(qos(:,1), qos(:,2:3:end), 'LineWidth', 2)
plot(qos(:,1), qos(:,2:3:end) - qos(:,4:3:end), 'LineWidth', 2);
plot([0 qos(end,1)], margin*ones(1,2), 'k', 'LineWidth', 2)
axis_limits = axis;
axis_limits([3 4]) = [0 0.2];
axis(axis_limits)

fid = fopen([filename '.csv'],'w'); 
fprintf(fid,['time,min_margin,confidence,margin1,std1,conf_interval1,'...
                                        'margin2,std2,conf_interval2,'...
                                        'margin3,std3,conf_interval3\n']);
fclose(fid);
dlmwrite([filename '.csv'],...
  [qos(:,1) [margin confidence].*ones(size(qos,1),2) qos(:,2:end)],'-append')

%% 3 task agents 3 network agent patrol (moving)

margin = 0.15;
confidence = 0.70;
filename = 'bags/3t3n_moving_paper';

panbag = rosbag([filename '.bag']);
qosbag = select(panbag, 'Topic', '/qos');
msg1 = readMessages(qosbag,1);
flows = length(msg1{1}.Data)/2;
qos = zeros(qosbag.NumMessages, flows*3+1);
qos(:,1) = qosbag.MessageList.Time;
qos(:,1) = qos(:,1) - qos(1,1);
for i = 1:qosbag.NumMessages
  msg = readMessages(qosbag,i);
  msg_data = msg{1}.Data';
  qos(i,2:3:end) = msg_data(1:2:end);
  qos(i,3:3:end) = sqrt(msg_data(2:2:end));
  qos(i,4:3:end) = norminv(confidence,0,1)*qos(i,3:3:end);
end

figure(2);clf;hold on
plot(qos(:,1), qos(:,2:3:end))
plot(qos(:,1), qos(:,2:3:end) - qos(:,4:3:end));
plot([0 qos(end,1)], margin*ones(1,2))
axis_limits = axis;
axis_limits([3 4]) = [0 0.4];
axis(axis_limits)

fid = fopen([filename '.csv'],'w'); 
fprintf(fid,['time,min_margin,confidence,margin1,std1,conf_interval1,'...
                                        'margin2,std2,conf_interval2,'...
                                        'margin3,std3,conf_interval3\n']);
fclose(fid);
dlmwrite([filename '.csv'],...
  [qos(:,1) [margin confidence].*ones(size(qos,1),2) qos(:,2:end)],'-append')

%% 3 task agents 3 network agent patrol (fixed)

margin = 0.15;
confidence = 0.70;
filename = 'bags/3t3n_fixed_paper';

panbag = rosbag([filename '.bag']);
qosbag = select(panbag, 'Topic', '/qos');
msg1 = readMessages(qosbag,1);
flows = length(msg1{1}.Data)/2;
qos = zeros(qosbag.NumMessages, flows*3+1);
qos(:,1) = qosbag.MessageList.Time;
qos(:,1) = qos(:,1) - qos(1,1);
for i = 1:qosbag.NumMessages
  msg = readMessages(qosbag,i);
  msg_data = msg{1}.Data';
  qos(i,2:3:end) = msg_data(1:2:end);
  qos(i,3:3:end) = sqrt(msg_data(2:2:end));
  qos(i,4:3:end) = norminv(confidence,0,1)*qos(i,3:3:end);
end

figure(3);clf;hold on
plot(qos(:,1), qos(:,2:3:end),'.')
plot(qos(:,1), qos(:,2:3:end) - qos(:,4:3:end),'.');
plot([0 qos(end,1)], margin*ones(1,2))
axis_limits = axis;
axis_limits([3 4]) = [0 0.4];
axis(axis_limits)

fid = fopen([filename '.csv'],'w');
fprintf(fid,['time,min_margin,confidence,margin1,std1,conf_interval1,'...
                                        'margin2,std2,conf_interval2,'...
                                        'margin3,std3,conf_interval3\n']);
fclose(fid);
dlmwrite([filename '.csv'],...
  [qos(:,1) [margin confidence].*ones(size(qos,1),2) qos(:,2:end)],'-append')

%% 3 task 6 network patrol (fixed)

margin = 0.15;
confidence = 0.70;
filename = 'bags/3task6network_fixed_paper';
out_name = 'bags/3task6network_fixed_paper';

panbag = rosbag([filename '.bag']);
qosbag = select(panbag, 'Topic', '/qos');
msg1 = readMessages(qosbag,1);
flows = length(msg1{1}.Data)/4;
qos = zeros(qosbag.NumMessages, flows*3+1);
qos(:,1) = qosbag.MessageList.Time;
qos(:,1) = qos(:,1) - qos(1,1);
for i = 1:qosbag.NumMessages
  msg = readMessages(qosbag,i);
  msg_data = msg{1}.Data';
  qos(i,2:3:end) = msg_data(2:4:end); % margin
  qos(i,3:3:end) = sqrt(msg_data(4:4:end)); % std
  qos(i,4:3:end) = norminv(confidence,0,1)*qos(i,3:3:end); % confidence interval
end

figure(4);clf;hold on
plot(qos(:,1), qos(:,2:3:end), 'LineWidth', 2)
plot(qos(:,1), qos(:,2:3:end) - qos(:,4:3:end), 'LineWidth', 2);
plot([0 qos(end,1)], margin*ones(1,2))
axis_limits = axis;
axis_limits([3 4]) = [0 0.4];
axis(axis_limits)

fid = fopen([out_name '.csv'],'w');
fprintf(fid,['time,min_margin,confidence,margin1,std1,conf_interval1,'...
                                        'margin2,std2,conf_interval2,'...
                                        'margin3,std3,conf_interval3\n']);
fclose(fid);
dlmwrite([out_name '.csv'],...
  [qos(:,1) [margin confidence].*ones(size(qos,1),2) qos(:,2:end)],'-append')

%% 3 task 6 network patrol (mobile)

margin = 0.15;
confidence = 0.70;
filename = 'bags/3task6network_moving_paper';
out_name = 'bags/3task6network_moving_paper';

panbag = rosbag([filename '.bag']);
qosbag = select(panbag, 'Topic', '/qos');
msg1 = readMessages(qosbag,1);
flows = length(msg1{1}.Data)/4;
qos = zeros(qosbag.NumMessages, flows*3+1);
qos(:,1) = qosbag.MessageList.Time;
qos(:,1) = qos(:,1) - qos(1,1);
for i = 1:qosbag.NumMessages
  msg = readMessages(qosbag,i);
  msg_data = msg{1}.Data';
  qos(i,2:3:end) = msg_data(2:4:end); % margin
  qos(i,3:3:end) = sqrt(msg_data(4:4:end)); % std
  qos(i,4:3:end) = norminv(confidence,0,1)*qos(i,3:3:end); % confidence interval
end

figure(5);clf;hold on
plot(qos(:,1), qos(:,2:3:end), 'LineWidth', 2)
plot(qos(:,1), qos(:,2:3:end) - qos(:,4:3:end), 'LineWidth', 2);
plot([0 qos(end,1)], margin*ones(1,2))
axis_limits = axis;
axis_limits([3 4]) = [0 0.4];
axis(axis_limits)

fid = fopen([out_name '.csv'],'w');
fprintf(fid,['time,min_margin,confidence,margin1,std1,conf_interval1,'...
                                        'margin2,std2,conf_interval2,'...
                                        'margin3,std3,conf_interval3\n']);
fclose(fid);
dlmwrite([out_name '.csv'],...
  [qos(:,1) [margin confidence].*ones(size(qos,1),2) qos(:,2:end)],'-append')
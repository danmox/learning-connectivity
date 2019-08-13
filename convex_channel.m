d = 0:0.1:40; % distance between agents

R1 = linear_channel(d);
R2 = sigmoid_channel(d);
R3 = stochastic_channel(d);

figure(1);clf;hold on;
plot(d,R1);
plot(d,R2);
plot(d,R3);
legend({'linear','sigmoid','stochastic'});
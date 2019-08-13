max_dist = 25; % max of linear model
agents = 3;

% set alphas
alpha_in = rand(agents,1);
alpha_in = rand*alpha_in/sum(alpha_in);
alpha_out = rand(agents,1);
alpha_out = rand*alpha_out/sum(alpha_out);
if sum(alpha_out) > sum(alpha_in)
  disp('alpha_out');
else
  disp('alpha_in');
end

agent = randi(agents,1); % agent which we are calculating the rate margin
alpha_in(agent) = 0; alpha_out(agent) = 0;

% define start and end positions
x0 = (max_dist+10)*rand(agents,1);
xf = (max_dist+10)*rand(agents,1);
v = xf - x0;

% margin
dv = 0:0.01:1;
margin = zeros(size(dv));
for i = 1:length(dv)
  d = x0 + v*dv(i);
  margin(i) = (alpha_out - alpha_in)'*linear_channel(d);
end

% plot results
plot(margin)
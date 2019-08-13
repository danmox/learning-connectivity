function R = sigmoid_channel(d)

R = 2 - 2 ./ (1 + exp(-d/9));

end
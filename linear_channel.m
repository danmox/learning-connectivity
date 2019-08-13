function R = linear_channel(d)

R = 1-d/25;
R(R < 0) = 0.0;

end
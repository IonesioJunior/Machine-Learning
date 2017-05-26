#!/usr/bin/octave -qf
arg_list = argv ();
data = load('data.txt');
y_axis = load('y_axis.txt');
theta = ones(size(data)(2),1);
alpha = str2num(arg_list{1});
it = str2num(arg_list{2});
sum_distance = [];

for i = 0:it
	hipotesys = data * theta;
	distance = hipotesys - y_axis;
	n_var = size(data)(2);
	sum_distance(end + 1) = (sum(distance(:)));
	for j = 1:n_var
		matrix = distance .* data(:,j);
		sum_result = (sum(matrix(:)) / size(data)(1)) * alpha;
		theta(j,1) = theta(j,1) - sum_result;
	end
end
	range = 0:1:it;
	hold on;
	plot(range,sum_distance);
	xlabel('Iterations');
	ylabel('Distance');
	title('Machine Learning');
	print -djpg image.jpg;
	sum_distance(end)
	theta

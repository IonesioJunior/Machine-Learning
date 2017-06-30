%Activation Function
function [result] = sigmoid(x)
	result = 1.0 ./ (1.0 + e .^ (-x));
end

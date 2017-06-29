function [result] = sigmoid(x)
	result = 1.0 ./ (1.0 + e .^ (-x));
end


function [THETA1_ACT,THETA2_ACT,JCost] = twoNeuronsHiden(X,Y,THETA1,THETA2,alpha,init=0,learn = 0)
	if(init == 1)
		THETA1 = 2 * rand(2,3) - 1;
		THETA2 = 2 *  rand(1,3) - 1;
	end
	T1_acc = zeros(size(THETA1));
	T2_acc = zeros(size(THETA2));
	m = 0;
	J = 0.0;
	for i = 1:rows(X)
		A1 = [1;X(i,1:end)'];
		Z2 = THETA1 * A1;
		A2 = [1 ; sigmoid(Z2)];
		Z3 = THETA2 * A2;
		hipotesys = sigmoid(Z3);
		m = m +1;
		%disp('Hypothesis for '), disp(X(i,1:2)), disp('is '), disp(hipotesys);		
		J = J + ( Y(i,1) * log(hipotesys) ) + ( (1 - Y(i,1)) * log(1 - hipotesys) );
	
		delta3 = hipotesys - Y(i,1);
		delta2 = ((THETA2' * delta3) .* (A2 .* (1 - A2)))(2:end);
		T2_acc = T2_acc + (delta3 * A2');
		T1_acc = T1_acc + (delta2 * A1');
	end
	
	THETA1 = THETA1 - (alpha * (T1_acc / m));
	THETA2 = THETA2 - (alpha * (T2_acc / m));
	THETA1_ACT = THETA1;
	THETA2_ACT = THETA2;
	JCost = J / -m;
end

function [result] = testAccuracy(X,Y,THETA1,THETA2)
	total = 0.0;
	correct = 0;
	for i = 1:rows(Y)
	        A1 = [1;X(i,1:end)'];
                Z2 = THETA1 * A1;
                A2 = [1 ; sigmoid(Z2)];
                Z3 = THETA2 * A2;
                hipotesys = sigmoid(Z3);
		if(hipotesys > 0.5)
			answer = 1;
		else
			answer = 0;
		end
		
		if(answer == Y(i,1))
			correct = correct + 1;
		end
	end
	result = (correct / rows(Y)) * 100.0;
end


arg_list = argv ();
data = load(arg_list{1});
alpha = str2num(arg_list{2});
X = data(1:size(data,1),1:size(data,2) - 1);
Y = data(1:size(data,1),size(data,2));
THETA1 = 0;
THETA2 = 0;
[THETA1,THETA2,JCost] = twoNeuronsHiden(X,Y,THETA1,THETA2,alpha,1,1);
for i = 1: 100000
	[THETA1,THETA2,JCost] = twoNeuronsHiden(X,Y,THETA1,THETA2,alpha,0,1);
	if(mod(i,1000) == 0)
		disp('Iteration: '),disp(i);
		disp(JCost);
		[result] = testAccuracy(X,Y,THETA1,THETA2);
		disp('Accuracy: '),disp(result);
	end
end

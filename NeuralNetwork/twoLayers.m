function [THETA1_ACT,THETA2_ACT,THETA3_ACT,JCost] = twoLayers(X,Y,THETA1,THETA2,THETA3,alpha,init=0)
	if(init == 1)
		THETA1 = 2 * rand(2,size(X,2) + 1) - 1;
		THETA2 = 2 * rand(2,size(X,2) + 1) - 1;
		THETA3 = 2* rand(1,size(X,2) + 1) - 1;
	end
	
	T1_acc = zeros(size(THETA1));
	T2_acc = zeros(size(THETA2));
	T3_acc = zeros(size(THETA3));
	J = 0.0;	
	for i = 1:rows(X)
		A1 = [1;X(i,1:end)'];
		Z2 = THETA1 * A1;
		A2 = [1;sigmoid(Z2)];
		Z3 = THETA2 * A2;
		A3 = [1;sigmoid(Z3)];
		Z4 = THETA3 * A3;
		hipotesys = sigmoid(Z4);
		
		J = J + ( Y(i,1) * log(hipotesys) ) + ( (1 - Y(i,1)) * log(1 - hipotesys) );
	
		delta4 = hipotesys - Y(i,1);
		delta3 = ((THETA3' * delta4) .* (A3 .* (1 - A3)))(2:end);
		delta2 = ((THETA2' * delta3) .* (A2 .* (1 - A2)))(2:end);
		T3_acc = T3_acc + (delta4 * A3');
		T2_acc = T2_acc + (delta3 * A2');
		T1_acc = T1_acc + (delta2 * A1');
	end
	m = size(X,1);
	
	THETA1 = THETA1 - (alpha * (T1_acc / m));
	THETA2 = THETA2 - (alpha * (T2_acc / m));
	THETA3 = THETA3 - (alpha * (T3_acc / m));
	THETA1_ACT = THETA1;
	THETA2_ACT = THETA2;
	THETA3_ACT = THETA3;
	JCost = J / -m;
end

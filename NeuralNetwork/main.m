% Simple Neural Network algorithm for binary classification
% author: Ionésio Junior
% Jun/29/2017
function [result] = testAccuracy(X,Y,THETA1,THETA2,THETA3 = 0)
	total = 0.0;
	correct = 0;
	for i = 1:rows(Y)
	        A1 = [1;X(i,1:end)'];
                Z2 = THETA1 * A1;
                A2 = [1 ; sigmoid(Z2)];
                Z3 = THETA2 * A2;
		if(THETA3 == 0)
                	hipotesys = sigmoid(Z3);
		else
			A3 = [1;sigmoid(Z3)];
			Z4 = THETA3 * A3;
			hipotesys = sigmoid(Z4);
		end


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

%Use one hiden layer to learn
function [result] = learnWithOneLayer(X,Y,alpha)
	THETA1 = 0;
	THETA2 = 0;
	[THETA1,THETA2,JCost] = oneLayer(X,Y,THETA1,THETA2,alpha,1,1);

	for i = 1: 100000
		[THETA1,THETA2,JCost] = oneLayer(X,Y,THETA1,THETA2,alpha,0,1);
		if(mod(i,1000) == 0) %Show Statistics progress every 1000 iterations
			disp('Iteration: '),disp(i);
			disp(JCost);
			[result] = testAccuracy(X,Y,THETA1,THETA2);
			disp('Accuracy: '),disp(result);
		end
	end
end

%Use two layers to learn
function [result] = learnWithTwoLayers(X,Y,alpha)
	THETA1 = 0;
	THETA2 = 0;
	THETA3 = 0;
	[THETA1,THETA2,THETA3,JCost] = twoLayers(X,Y,THETA1,THETA2,THETA3,alpha,1);
	for i = 1:10000
		[THETA1,THETA2,THETA3,JCost] = twoLayers(X,Y,THETA1,THETA2,THETA3,alpha,0);
                if(mod(i,1000) == 0) %Show Statistics progress every 1000 iterations
                        disp('Iteration: '),disp(i);
                        disp(JCost);
                        [result] = testAccuracy(X,Y,THETA1,THETA2,THETA3);
                        disp('Accuracy: '),disp(result);
                end
	end
end


arg_list = argv ();
data = load(arg_list{1});
alpha = str2num(arg_list{2});

%Split features and ground truth
X = data(1:size(data,1),1:size(data,2) - 1);
Y = data(1:size(data,1),size(data,2));
learnWithTwoLayers(X,Y,alpha);

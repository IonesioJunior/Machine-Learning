
function [Raiz, Iter, Info,values] = newton_raphson(funcao, derivada, x0, Toler, IterMax, Exibe)
	x = x0;
	Iter = 0;
	deltax = 1;
	if Exibe
		fprintf('\n Cálculo de raiz de equacao pelo método de Newton-Raphson\n\n')
		fprintf(' k    x_k    Fx_k    DFx_k   deltax_k\n')
	end
	values = [];
	while 1
		Fx = eval(funcao);DFx = eval(derivada);
		if Exibe
			fprintf('%3i%11.5f%14.5e%14.4e',Iter,x,Fx,DFx);
		end
		if (abs(deltax) <= Toler && abs(Fx) <= Toler) || DFx == 0 || Iter >= IterMax
			break
		end
		values = [values [Fx;x]];
		deltax = Fx / DFx;
		x = x - deltax;
		Iter = Iter + 1;
		if Exibe
			fprintf('%14.5e\n',deltax)
		end
	end
	values = double(values)
	Raiz = x;
	Info = abs(deltax) > Toler || abs(Fx) > Toler;
	if Exibe
		fprintf('\n\nRaiz = %9.5f\nIter = %3i\nInfo = %3i\n',Raiz,Iter,Info)
	end


[r,it,in,values] = newton_raphson('9*(10^9) * ( ( (2*10^-3) * (2*10^-3) ) / (x^2))','-72000/x^3',0.01,1e-10,10,0);
plotData(values(2,:),values(1,:),'Força Eletroestática','Distancia','Força x Distancia')


[r,it,in,values] = newton_raphson('9*(10^9) * (((2*10^-3) * (x * 10^-3) ) / (0.2^2))','450000',5,1e-10,10,0);
plotData(values(2,:),values(1,:),'Força Eletroestática','Carga','Força x Carga')

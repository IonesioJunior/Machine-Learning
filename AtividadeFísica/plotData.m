function plotData(X,y,y_label,x_label,plotTitle)
	figure;
	ylabel(y_label);
	xlabel(x_label);
	title(plotTitle);
	plot(X,y);
	pause;
end

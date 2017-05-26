#coding: utf-8
from random import randint

theta_zero = randint(0,50)
theta_one = randint(0,50)
theta_two = randint(0,50)

def func_generator(x,z):
	return theta_zero + (theta_one * x) + (theta_two * (z ** 2))

data = open('data.txt','w')
y_values = []
for i in range(10):
	x_value = randint(1,50)
	z_value = randint(1,50)
        y_values.append(func_generator(x_value,z_value))
	print str(theta_zero) + " + " + str(theta_one) + "*" + str(x_value) + " + " + str(theta_two) + " * " + str(z_value) + "^2 = " + str(y_values[i])
	data.write("1 " + str(x_value)+ " " + str(z_value) + "\n")
data.close()

new_data = open('y_axis.txt','w')
for i in range(len(y_values)):
	new_data.write(str(y_values[i]) + "\n")
new_data.close()


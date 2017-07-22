import sys
import theano
import theano.tensor as T
import numpy
import cPickle


class LogisticRegression(object):
	
	
	def __init__(self,input,n_input,n_output):
		
		# Definindo a matrix de pesos
		self.W = theano.shared(
			value = numpy.zeros(
				(n_input,n_output), #Dimensoes da matriz de pesos no formato de matriz numpy
				dtype = theano.config.floatX # Tipo de dado
			),
			name = 'W',
			borrow = True
		)
		
		# Definindo o vetor de bias
		self.b = theano.shared(
			value = numpy.zeros(
				(n_output,), # Dimensoes do vetor de bias no formato numpy
				dtype = theano.config.floatX # tipo de dado armazenado no vetor
			),
			name = 'b',
			borrow = True
		)
		
		# Acredito que seja a hipotese dos dados de entrada * a matriz de pesos + bias
		self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W) + self.b)
		
		#Computa a classe com maxima probabilidade
		self.y_pred = T.argmax(self.p_y_given_x,axis= 1)
		
		self.params = [self.W,self.b]
		self.input = input

	
	def negative_log_likelihood(self,y):
		# Calcula o custo e a diferenca entre a hipotese e os resultados reais
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
	
	
	def errors(self,y):
		return T.mean(T.neq(self.y_pred,y))
	


def shared_dataset(data_xy,borrow = True):
	data_x,data_y = data_xy
	shared_x = theano.shared(
					numpy.asarray(data_x,dtype=theano.config.floatX),
					borrow = True
				)
	
	shared_y = theano.shared(
					numpy.asarray(data_y,dtype= theano.config.floatX),
					borrow = True
				)
	return shared_x, T.cast(shared_y,'int32')



def load_data(data_set):
	with open(data_set,'rb') as f:
		try:
			train_set , valid_set , test_set = cPickle.load(f)
		except:
			raise Exception("Can't load this file!")
	
	test_set_x , test_set_y = shared_dataset(test_set)
	valid_set_x , valid_set_y = shared_dataset(valid_set)
	train_set_x , train_set_y = shared_dataset(train_set)
	rval = [ (train_set_x,train_set_y) , (valid_set_x,valid_set_y) , (test_set_x,test_set_y)]
	return rval




def gradientDescent(dataset,lr = 0.1,n_epoch = 1000,batch_size = 600):
	
	datasets = load_data(dataset)
	train_set_x , train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x,test_set_y = datasets[2]
	
	
	# Pegando o numero total de batches fazendo a divisao do numero total de amostras e dividindo pelo tamanho de um batch
	n_train_batches = train_set_x.get_value(borrow = True).shape[0] // batch_size
	n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] // batch_size
	n_test_batches = test_set_x.get_value(borrow = True).shape[0] // batch_size
	
	
	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')
	
	
	classifier = LogisticRegression(input=x,n_input=28 * 28,n_output = 10)
	cost = classifier.negative_log_likelihood(y) # Custo inicial
	
	g_W = T.grad(cost=cost,wrt=classifier.W)
	g_b = T.grad(cost=cost,wrt=classifier.b)
	
	updates = [ 
		    (classifier.W,classifier.W - lr * g_W),
		    (classifier.b,classifier.b - lr * g_b)	
		  ]


	#Definicao dos modelos em um batch
	train_model = theano.function(
					inputs = [index],
					outputs = cost,
					updates = updates ,
					givens = {
							x : train_set_x[index * batch_size : (index + 1) * batch_size ],
							y : train_set_y[index * batch_size : (index + 1) * batch_size ]
						 }
				     )
	
	validate_model = theano.function(
					  inputs = [index],
					  outputs = classifier.errors(y),
					  givens = {
							x : valid_set_x[index * batch_size : (index + 1) * batch_size ],
							y : valid_set_y[index * batch_size : (index + 1) * batch_size ]
						   }
					)
	
	test_model = theano.function(
					inputs = [index],
					outputs = classifier.errors(y),
					givens = {
							x : test_set_x[index * batch_size : (index + 1) * batch_size ],
							y : test_set_y[index * batch_size : (index + 1) * batch_size ]
						 }
				    )
	
	epoch = 0
	patience = 5000
	validation_frequency = patience // 2
	best_validation_loss = numpy.inf
	while epoch < n_epoch :
		epoch = epoch + 1
		minibatch_avg_cost = 0
		for minibatch_index in range(n_train_batches):
			minibatch_avg_cost += train_model(minibatch_index)
			iteration = (epoch - 1) *  n_train_batches + minbatch_index
			if iteration % validation_frequency == 0:
				validation_loss = [validate_model(i) for i in range(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_loss)
				print "Validation Error : %f %%" % (this_validation_loss * 100)
				
				if this_validation_loss < best_validation_loss:
					best_validation_loss = this_validation_loss
	
				with open('best_model.pkl','wb') as f:
					pickle.dump(classifier,f)
	
		if epoch % 100 == 0:
			test_loss = [test_model(i) for i in range(n_test_batches)]
			this_test_loss = numpy.mean(test_loss)
	

if __name__ == "__main__":
	gradientDescent(sys.argv[1])

#coding: utf-8
import os
import sys
sys.path.append("../LogisticRegression/")
import numpy
import theano
import theano.tensor as T


from logisticRegression import LogisticRegression, load_data


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,activation=T.tanh):
        '''
        Função de ativação é dada por: tanh(dot(input,W) + b)

	Params: 

        rng (numpy.random.RandomState) : Gerador de numeros aleatorios para definir o valor inical dos pesos em W
	input (theano.tensor.dmatrix) : Variavel simbólica de tamanho (n_exemplos,n_input)
        n_in (Int) : dimensionality of input
        n_out (Int) : Número de unidades na camada oculta
	activation (function) : função nao linear aplicada na camada oculta
        '''

        self.input = input

        # Inicialização dos pesos usando a formula:
        # sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

	#Inicialização dos valores de bias
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

	#Acredito que seja a operação de multiplicação entre os exemplos  * matriz de pesos 
        lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None #Caso nao exista
            else activation(lin_output) #Caso exista função de ativação
        )
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Inicializa os parametros da rede perceptron

	Params:

        rng (numpy.random.RandomState) : Gerador de numeros aleatorios para definir o valor inical dos pesos em W
        input (theano.tensor.TensorType) : Variavel simbólica para armazenar a quantidade de exemplos de um minibatch
        n_in (Int) : numero de caracteristicas a serem usadas pela rede
	n_hidden (Int) : número de camadas ocultas
        n_out (Int) : numero de unidades de saída ( classes possiveis )
        """

        # Inicialização de uma camada oculta
	self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh # Tipo da função de ativação
        )

	# Ultima Layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output, # Pega o resultado gerado pela camada oculta e usa como entrada na camada de regressão
            n_input=n_hidden,
            n_output=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        # Calculo dos erros
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
	
        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,dataset='mnist.pkl', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Definindo a quantidade de batches para treino , validação e teste
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # Determina variaveis simbólicas  x e y
    index = T.lscalar()  # indice do minibatch
    x = T.matrix('x')  # vetor com o valor dos pixels da imagem
    y = T.ivector('y')  # Representa um vetor com os labels das classes

    rng = numpy.random.RandomState(1234)

    # O classifier dessa vez é constituido por um objeto Mult Layer Perceptron ao invés de apenas uma regressão logística
    classifier = MLP(
        rng=rng,
        input=x, # variavel simbolica que determina o minibatch que será usado
        n_in=28 * 28, # vetor com 28 * 28 valores (cada um representando um pixel)
        n_hidden=n_hidden, # numero de camadas ocultas
        n_out=10 # 10 labels de saída
    )

    # O custo para minimizar os erros do treinamento é dado pelo valor do negative_log_likelihood() + (regularização de L1 * L1) + (regularização de L2 * L2)
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # Calcula os erros para o conjunto de testes
    test_model = theano.function(
        inputs=[index], # Determina o intervalo de exemplos em que serão computados os erros
        outputs=classifier.errors(y), # A saída é o valor quantitativo do erro para aquele minibatch gerado pela mlp
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size], # Determina um vetor de exemplos equivalente a 1 minibatch
            y: test_set_y[index * batch_size:(index + 1) * batch_size] # Determina um vetor de labels equivalente a 1 minibatch
        }
    )

    # Calcula os erros para o conjunto de validação
    validate_model = theano.function(
        inputs=[index], # Determina o intervalo de exemplos em que serão computados os erros
        outputs=classifier.errors(y), # A saíde é o valor quantitativo do erro para aquele minibatch gerado pela mlp
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size], # O conjunto de exemplos que será fornecido para executar os testes
            y: valid_set_y[index * batch_size:(index + 1) * batch_size] # O conjunto de labels que será fornecido para computar os erros entre o predicted e o label real
        }
    )


    # Computa o valor do custo do gradient para cada parametro e armazena no vetor
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # Definição do modelo de treino da rede
    train_model = theano.function(
        inputs=[index], # Indice do minibatch a ser passado para treino
        outputs=cost, # Retorna o custo daquele minibatch em especifico
        updates=updates, # Faz o update dos parametros de acordo com o custo do gradient para aquele conjunto de treino
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size], # Determina o conjunto de exemplos que será usado para treinar nessa iteração
            y: train_set_y[index * batch_size: (index + 1) * batch_size] # Determina o conjunto de labels que será usado para treinar nessa iteração
        }
    )

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

if __name__ == '__main__':
    test_mlp()

from __future__ import print_function, division
import os
import sys
import timeit

import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.decomposition import PCA

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from dataset_location import *


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that).
    return shared_x, shared_y


def load_data(dataset, pca=2):
    """ The load dataset function
    
    This function covers for singular dataset
    (either DNA Methylation, Gene Expression, or miRNA Expression)
    for survival rate prediction.
    Input is .npy file location in string format.
    Output is in Theano shared variable format
    to speed up computation with GPU.
    """

    # Initialize list of dataset files' name
    temp_input = []

    # Input list of dataset files' name
    # I. Gene + miRNA
    if (dataset==7) or (dataset==8) or (dataset==9):
        n_dataset = 2
        # 1. Gene Expression
        if (dataset==7):    # 1.a. Gene htsec-count
            temp_input.append(INPUT_GEN_GEN_MIR_SURVIVAL_COUNT)
        elif (dataset==8):  # 1.b. Gene htsec-FPKM
            temp_input.append(INPUT_GEN_GEN_MIR_SURVIVAL_FPKM)
        elif (dataset==9):  # 1.c. Gene htsec-FPKMUQ
            temp_input.append(INPUT_GEN_GEN_MIR_SURVIVAL_FPKMUQ)
        # 2. miRNA Expression
        temp_input.append(INPUT_MIR_GEN_MIR_SURVIVAL)
        # Labels
        temp_label = LABELS_GEN_MIR_SURVIVAL
    
    # II. Methylation + Gene + miRNA
    elif (dataset==10) or (dataset==12) or (dataset==14):
        n_dataset = 3
        # 1. Methylation Platform GPL8490  (27578 cpg sites)
        temp_input.append(INPUT_MET_MET_GEN_MIR_SURVIVAL)
        # 2. Gene Expression
        if (dataset==10):    # 2.a. Gene htsec-count
            temp_input.append(INPUT_GEN_MET_GEN_MIR_SURVIVAL_COUNT)
        elif (dataset==12):  # 2.b. Gene htsec-FPKM
            temp_input.append(INPUT_GEN_MET_GEN_MIR_SURVIVAL_FPKM)
        elif (dataset==14):  # 2.c. Gene htsec-FPKMUQ
            temp_input.append(INPUT_GEN_MET_GEN_MIR_SURVIVAL_FPKMUQ)
        # 3. miRNA Expression
        temp_input.append(INPUT_MIR_MET_GEN_MIR_SURVIVAL)
        # Labels
        temp_label = LABELS_MET_GEN_MIR_SURVIVAL
    
    # III. Methylation (Long) + Gene + miRNA
    elif (dataset==11) or (dataset==13) or (dataset==15):
        n_dataset = 3
        # 1. Methylation Platform GPL16304 (485577 cpg sites)
        temp_input.append(INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL)
        # 2. Gene Expression
        if (dataset==11):    # 2.a. Gene htsec-count
            temp_input.append(INPUT_GEN_METLONG_GEN_MIR_SURVIVAL_COUNT)
        elif (dataset==13):  # 2.b. Gene htsec-FPKM
            temp_input.append(INPUT_GEN_METLONG_GEN_MIR_SURVIVAL_FPKM)
        elif (dataset==15):  # 2.c. Gene htsec-FPKMUQ
            temp_input.append(INPUT_GEN_METLONG_GEN_MIR_SURVIVAL_FPKMUQ)
        # 3. miRNA Expression
        temp_input.append(INPUT_MIR_METLONG_GEN_MIR_SURVIVAL)
        # Labels
        temp_label = LABELS_METLONG_GEN_MIR_SURVIVAL

    
    min_max_scaler = MinMaxScaler()     # Initialize normalization function
    rval = []                           # Initialize list of outputs

    # Iterate for the number of dataset
    for j in range(n_dataset):
        # Load the dataset as 'numpy.ndarray'
        try:
            input_set = numpy.load(temp_input[j])
            label_set = numpy.load(temp_label)
        except Exception as e:
            sys.exit("Change your choice of features because the data is not available")

        # feature selection by PCA
        if pca == 1:
            pca0 = PCA(n_components=600)
            input_set = pca0.fit_transform(input_set)

        # normalize input
        input_set = min_max_scaler.fit_transform(input_set)

        rval.extend((input_set, label_set))

    return rval


class LinearRegression(object):
    """ Linear Regression class
    
    Linear regression consist of 1 input layer and 1 output layer.
    It functions as the last 2 layers of the DBN.
    The output layer uses identity function as the activation function.
    The cost function is mean square error function.
    """

    def __init__(self, input, n_in, dropout=0.):
        """ Linear Regression initialization function

        Linear Regression is defined by input of Theano tensor matrix variable and size of input layer.
        Linear regression parameters (weight and bias) are created based on these.
        The predicted output uses identity function as activation function.
        """
        self.input = input

        self.W = theano.shared(value=numpy.zeros((n_in), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=0., name='b', borrow=True)
        self.params = [self.W, self.b]

        srng = MRG_RandomStreams()

        # Dropout
        retain_prob = 1 - dropout
        input *= srng.binomial(input.shape, p=retain_prob, dtype=theano.config.floatX)
        input /= retain_prob

        self.p_y_given_x = T.dot(input, self.W) + self.b        

    def mean_square_error(self, y):
        """ Linear Regression mean square error cost function

        The cost function is calculated using the predicted output and the actual output.
        """
        return T.mean(T.sqr(self.p_y_given_x - y))
        
    def input_last_layer(self):
        """ Logistic Regression input function """
        return self.input
        
    def y_predict(self):
        """ Linear Regression predicted output function """
        return self.p_y_given_x


class HiddenLayer(object):
    """ Hidden Layer class
    
    It defines a hidden layer with adjustable activation function.
    """
    
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """ Hidden Layer initialization function

        Hidden Layer is defined by input of Theano tensor matrix variable,
        size of input layer, and size of output layer.
        Hidden layer parameters (weight and bias) are created based on these.
        """
        self.input = input
        
        if W is None:
            W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        self.params = [self.W, self.b]


class RBM(object):
    """ RBM class
    
    It implements either Contrastive Divergence (CD) or Persistent Contrastive Divergence (PCD).
    """

    def __init__(self, input=None, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None, numpy_rng=None, theano_rng=None):
        """ RBM initialization function

        Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing Contrastive Divergence updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            initial_W = numpy.asarray(numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)), high=4 * numpy.sqrt(6. / (n_hidden + n_visible)), size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            hbias = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), name='hbias', borrow=True)

        if vbias is None:
            vbias = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX), name='vbias', borrow=True)

        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]
        
    def free_energy(self, v_sample):
        """ Free Energy function """
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        """ Propup function 

        This function propagates the visible units activation upwards to
        the hidden units
        """
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        """ Sample H given V function 

        This function propagates the visible units activation upwards to
        the hidden units, then take a sample of the hidden units given
        their activation functions.
        
        For GPU usage, specify theano_rng.binomial to return the dtype floatX
        """
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        """ Propup function 

        This function propagates the hidden units activation downwards to
        the visible units
        """
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        """ Sample H given V function 

        This function propagates the hidden units activation downwards to
        the visible units, then take a sample of the visible units given
        their activation functions.
        
        For GPU usage, specify theano_rng.binomial to return the dtype floatX
        """
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """ Gibbs HVH function 

        This function performs one step of Gibbs sampling starting from the hidden state
        """
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """ Gibbs VHV function 

        This function performs one step of Gibbs sampling starting from the visible state
        """
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """ Get Cost Updates function

        This functions implements one step of Contrastive Divergence (CD)
        or Persistent Contrastive Divergence (PCD)

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns monitoring cost and the updated dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if PCD is used.
        """

        # compute positive phase for CD
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # determine start of the chain
        # CD uses newly generated hidden state
        # PCD uses the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        
        # Gibbs sampling for k steps to find the end of the chain
        ([pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates) = theano.scan(self.gibbs_hvh, outputs_info=[None, None, None, None, None, chain_start], n_steps=k, name="gibbs_hvh")
        chain_end = nv_samples[-1]

        # Cost function and parameter updates
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
        
        # Monitoring cost
        # For PCD, update the persistent variable with the end of current chain
        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates
        
    def get_pseudo_likelihood_cost(self, updates):
        """ Pseudo Likelihood Cost function 

        Monitoring cost for PCD.
        """

        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        xi = T.round(self.input)

        fe_xi = self.free_energy(xi)

        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)

        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """ Reconstruction Cost function 

        Monitoring cost for CD.
        """
        cross_entropy = T.mean(T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) + (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)), axis=1))
        return cross_entropy


class DBN(object):
    """ DBN class
    
    DBN consist of 1 input layer, at least 1 of hidden layer, and 1 output layer.
    The initialization define the rbm and sigmoid layer as the hidden layer, and a linear regression as the output layer
    The output layer uses softmax as the activation function.
    The cost function is mean square error function.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784, hidden_layers_sizes=[500, 500], n_outs=1):
        """ DBN initialization function

        DBN is defined by size of input layer and hidden layers.
        This function defines the 
        DBN parameters (weight and bias) are created based on these.
        The predicted output uses identity function as activation function.
        Predicted label is neuron in output layer with highest value.
        """
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.fmatrix('x')
        self.y = T.fvector('y')
        self.dropout = T.dscalar('dropout')
        
        # Iterate for as many numbers of hidden layers
        # So, the size of sigmoid_layers and rbm_layers is the same as the size of hidden layers
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            # Sigmoid hidden layers
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                input=layer_input,
                n_in=input_size,
                n_out=hidden_layers_sizes[i],
                activation=T.nnet.sigmoid)

            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

            # RBM hidden layers
            # RBM shares its weights and hidden biases with the sigmoid layers
            rbm_layer = RBM(numpy_rng=numpy_rng,
                theano_rng=theano_rng,
                input=layer_input,
                n_visible=input_size,
                n_hidden=hidden_layers_sizes[i],
                W=sigmoid_layer.W,
                hbias=sigmoid_layer.b)
            
            self.rbm_layers.append(rbm_layer)

        # Linear Regression output layer
        self.linLayer = LinearRegression(input=self.sigmoid_layers[-1].output,
                                         n_in=hidden_layers_sizes[-1],
                                         dropout=self.dropout)

        self.params.extend(self.linLayer.params)

        # cost function
        self.finetune_cost = self.linLayer.mean_square_error(self.y)

        # predicted input of last layer function
        self.input_last_layer = self.linLayer.input_last_layer()

        # error function
        self.y_predict = self.linLayer.y_predict()

    def pretraining_functions(self, train_set_x, batch_size, k):
        """ DBN Pretraining function

        It implements series of RBMs.
        The default is using CD (persistent=None).
        The output is series of RBM functions.
        Each function updates paratemers on each RBM and outputs a monitoring cost.
        """
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            cost, updates = rbm.get_cost_updates(learning_rate, persistent=None, k=k)

            fn = theano.function(inputs=[index, theano.In(learning_rate, value=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:batch_end]})
            pretrain_fns.append(fn)

        return pretrain_fns

    def pretraining_bottom_layer_output(self, train_set_x, dropout=0.):
        """ Predict function

        Predict the output of the DBN level-1
        """

        index = T.lscalar('index')

        # function
        train_score_i = theano.function([index],
            self.input_last_layer,
            on_unused_input='ignore',
            givens={self.x: train_set_x[index:],
                    self.dropout: 0.})

        def train_score():
            return train_score_i(0)

        return train_score

    def build_finetune_functions(self, train_set_x, train_set_y, batch_size, learning_rate, dropout=0., optimizer=1):
        """ DBN Finetune function

        Implement train function
        All functions done per batch size
        Train function results in the finetune cost (mean square error cost) and parameter update
        """

        index = T.lscalar('index')

        # Parameter update
        def SGD(cost, params, lr=0.0002):
            updates = []
            grads = T.grad(cost, params)
            for p, g in zip(params, grads):
                updates.append((p, p - g * lr))
            return updates

        def RMSprop(cost, params, lr=0.0002, rho=0.9, epsilon=1e-6):
            grads = T.grad(cost=cost, wrt=params)
            updates = []
            for p, g in zip(params, grads):
                acc = theano.shared(p.get_value() * 0.)
                acc_new = rho * acc + (1 - rho) * g ** 2
                gradient_scaling = T.sqrt(acc_new + epsilon)
                g = g / gradient_scaling
                updates.append((acc, acc_new))
                updates.append((p, p - lr * g))
            return updates

        def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
            updates = []
            grads = T.grad(cost, params)
            i = theano.shared(numpy.asarray(0., dtype=theano.config.floatX))
            i_t = i + 1.
            fix1 = 1. - (1. - b1)**i_t
            fix2 = 1. - (1. - b2)**i_t
            lr_t = lr * (T.sqrt(fix2) / fix1)
            for p, g in zip(params, grads):
                m = theano.shared(p.get_value() * 0.)
                v = theano.shared(p.get_value() * 0.)
                m_t = (b1 * g) + ((1. - b1) * m)
                v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
                g_t = m_t / (T.sqrt(v_t) + e)
                p_t = p - (lr_t * g_t)
                updates.append((m, m_t))
                updates.append((v, v_t))
                updates.append((p, p_t))
            updates.append((i, i_t))
            return updates

        if optimizer == 1:
            updates = SGD(cost=self.finetune_cost, params=self.params, lr=learning_rate)
        elif optimizer == 2:
            updates = RMSprop(cost=self.finetune_cost, params=self.params, lr=learning_rate)
        elif optimizer == 3:
            updates = Adam(cost=self.finetune_cost, params=self.params, lr=learning_rate)

        # train function
        train_fn = theano.function(inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    self.y: train_set_y[index * batch_size: (index + 1) * batch_size],
                    self.dropout: dropout})

        return train_fn

    def predict(self, test_set_x, dropout=0.):
        """ Predict function

        Predict the output of the test input
        """

        index = T.lscalar('index')

        # test function
        test_score_i = theano.function([index],
            self.y_predict,
            on_unused_input='ignore',
            givens={self.x: test_set_x[index:],
                    self.dropout: 0.})

        def test_score():
            return test_score_i(0)

        return test_score


class mDBN(object):
    """ mDBN class
    
    mDBN consist of at least two different data types.
    This class implement training function for mDBN and output prediction function.
    """

    def __init__(self, numpy_rng, n_ins, hidden_layers_lvl1_sizes, hidden_layers_lvl2_sizes, n_outs, W_lvl1, b_lvl1, W_lvl2, b_lvl2):
        """ mDBN initialization function

        mDBN is defined by size of input layer, hidden layers, output layer.
        This function defines the 
        mDBN parameters (weight and bias) are created based on these.
        The predicted output uses softmax as activation function.
        Predicted label is neuron in output layer with highest value.
        """
        n_datasets = len(hidden_layers_lvl1_sizes)
        self.params = []
        self.sigmoid_layers_lvl1 = []
        self.sigmoid_layers_lvl2 = []
        
        self.x0 = T.fmatrix('x0')
        self.x1 = T.fmatrix('x1')
        self.x2 = T.fmatrix('x2')
        
        if n_datasets == 2:
            self.xs = [self.x1,self.x2]
        elif n_datasets == 3:
            self.xs = [self.x0,self.x1,self.x2]
        
        for j in range(n_datasets):
            sig_layers = []
            self.sigmoid_layers_lvl1.append(sig_layers)

        self.y = T.fvector('y')
        self.dropout = T.dscalar('dropout')
        
        # Forward propagation
        for dataset in range(n_datasets):
            # Iterate for as many numbers of hidden layers
            # So, the size of sigmoid_layers is the same as the size of hidden layers
            for i in range(len(hidden_layers_lvl1_sizes[dataset])):
                if i == 0:
                    input_size = n_ins[dataset]
                else:
                    input_size = hidden_layers_lvl1_sizes[dataset][i - 1]

                if i == 0:
                    layer_input = self.xs[dataset]
                else:
                    layer_input = self.sigmoid_layers_lvl1[dataset][-1].output

                # Sigmoid hidden layers
                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                    input=layer_input,
                    n_in=input_size,
                    n_out=hidden_layers_lvl1_sizes[dataset][i],
                    W = W_lvl1[dataset][i],
                    b = b_lvl1[dataset][i],
                    activation=T.nnet.sigmoid)

                self.sigmoid_layers_lvl1[dataset].append(sigmoid_layer)

                self.params.extend(sigmoid_layer.params)

        
        # Iterate for as many numbers of hidden layers
        # So, the size of sigmoid_layers and rbm_layers is the same as the size of hidden layers
        for i in range(len(hidden_layers_lvl2_sizes)):
            if i == 0:
                temp_n_ins = 0
                for k in range(n_datasets):
                    temp_n_ins = temp_n_ins + hidden_layers_lvl1_sizes[k][-1]
                input_size = temp_n_ins
            else:
                input_size = hidden_layers_lvl2_sizes[i - 1]

            if i == 0:
                x_lvl2 = self.sigmoid_layers_lvl1[0][-1].output
                for k in range(n_datasets-1):
                    x_lvl2 = T.concatenate([x_lvl2, self.sigmoid_layers_lvl1[k+1][-1].output], axis=1)
                layer_input = x_lvl2
            else:
                layer_input = self.sigmoid_layers_lvl2[-1].output

            # Sigmoid hidden layers
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                input=layer_input,
                n_in=input_size,
                n_out=hidden_layers_lvl2_sizes[i],
                W = W_lvl2[i],
                b = b_lvl2[i],
                activation=T.nnet.sigmoid)

            self.sigmoid_layers_lvl2.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)


        # Logistic Regression output layer
        self.linLayer = LinearRegression(input=self.sigmoid_layers_lvl2[-1].output,
                                           n_in=hidden_layers_lvl2_sizes[-1],
                                           dropout=self.dropout)
        
        self.params.extend(self.linLayer.params)

        # cost function
        self.finetune_cost = self.linLayer.mean_square_error(self.y)

        # predicted output function
        self.y_predict = self.linLayer.y_predict()

    
    def build_finetune_functions(self, train_sets_x, train_set_y, batch_size, learning_rate, dropout=0., optimizer=1):
        """ DBN Finetune function

        Implement train function
        All functions done per batch size
        Train function results in the finetune cost (negative log likelihood cost) and parameter update
        """

        index = T.lscalar('index')

        # Parameter update
        def SGD(cost, params, lr=0.0002):
            updates = []
            grads = T.grad(cost, params)
            for p, g in zip(params, grads):
                updates.append((p, p - g * lr))
            return updates

        def RMSprop(cost, params, lr=0.0002, rho=0.9, epsilon=1e-6):
            grads = T.grad(cost=cost, wrt=params)
            updates = []
            for p, g in zip(params, grads):
                acc = theano.shared(p.get_value() * 0.)
                acc_new = rho * acc + (1 - rho) * g ** 2
                gradient_scaling = T.sqrt(acc_new + epsilon)
                g = g / gradient_scaling
                updates.append((acc, acc_new))
                updates.append((p, p - lr * g))
            return updates

        def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
            updates = []
            grads = T.grad(cost, params)
            i = theano.shared(numpy.asarray(0., dtype=theano.config.floatX))
            i_t = i + 1.
            fix1 = 1. - (1. - b1)**i_t
            fix2 = 1. - (1. - b2)**i_t
            lr_t = lr * (T.sqrt(fix2) / fix1)
            for p, g in zip(params, grads):
                m = theano.shared(p.get_value() * 0.)
                v = theano.shared(p.get_value() * 0.)
                m_t = (b1 * g) + ((1. - b1) * m)
                v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
                g_t = m_t / (T.sqrt(v_t) + e)
                p_t = p - (lr_t * g_t)
                updates.append((m, m_t))
                updates.append((v, v_t))
                updates.append((p, p_t))
            updates.append((i, i_t))
            return updates

        if optimizer == 1:
            updates = SGD(cost=self.finetune_cost, params=self.params, lr=learning_rate)
        elif optimizer == 2:
            updates = RMSprop(cost=self.finetune_cost, params=self.params, lr=learning_rate)
        elif optimizer == 3:
            updates = Adam(cost=self.finetune_cost, params=self.params, lr=learning_rate)

        # train function
        if len(train_sets_x) == 2:
            train_set_x_1, train_set_x_2 = train_sets_x
            train_fn = theano.function(inputs=[index], outputs=self.finetune_cost, updates=updates,
                                       givens={self.x1: train_set_x_1[index * batch_size: (index + 1) * batch_size],
                                               self.x2: train_set_x_2[index * batch_size: (index + 1) * batch_size],
                                               self.y: train_set_y[index * batch_size: (index + 1) * batch_size],
                                               self.dropout: dropout})
        elif len(train_sets_x) == 3:
            train_set_x_0, train_set_x_1, train_set_x_2 = train_sets_x
            train_fn = theano.function(inputs=[index], outputs=self.finetune_cost, updates=updates,
                                       givens={self.x0: train_set_x_0[index * batch_size: (index + 1) * batch_size],
                                               self.x1: train_set_x_1[index * batch_size: (index + 1) * batch_size],
                                               self.x2: train_set_x_2[index * batch_size: (index + 1) * batch_size],
                                               self.y: train_set_y[index * batch_size: (index + 1) * batch_size],
                                               self.dropout: dropout})

        return train_fn

    def predict(self, test_sets_x, dropout=0.):
        """ Predict function

        Predict the output of the test input
        """

        index = T.lscalar('index')

        # test function
        if len(test_sets_x) == 2:
            test_set_x_1, test_set_x_2 = test_sets_x
            test_score_i = theano.function([index], self.y_predict, on_unused_input='ignore',
                                           givens={self.x1: test_set_x_1[index:],
                                                   self.x2: test_set_x_2[index:],
                                                   self.dropout: 0.})
        elif len(test_sets_x) == 3:
            test_set_x_0, test_set_x_1, test_set_x_2 = test_sets_x
            test_score_i = theano.function([index], self.y_predict, on_unused_input='ignore',
                                           givens={self.x0: test_set_x_0[index:],
                                                   self.x1: test_set_x_1[index:],
                                                   self.x2: test_set_x_2[index:],
                                                   self.dropout: 0.})

        def test_score():
            return test_score_i(0)

        return test_score


def test_mDBN(finetune_lr=0.1,
    pretraining_epochs=100,
    pretrain_lr=0.01,
    k=1,
    training_epochs=100,
    dataset=7,
    batch_size=10,
    layers_met=[1000, 1000, 1000],
    layers_gen=[1000, 1000, 1000],
    layers_mir=[1000, 1000, 1000],
    layers_tot=[500, 500, 500],
    dropout=0.2,
    pca=2,
    optimizer=1):
    
    # Title
    print("\nSurvival Rate Regression with ", end="")
    if (dataset==10) or (dataset==12) or (dataset==14):
        print("DNA Methylation Platform GPL8490, ", end="")
    if (dataset==11) or (dataset==13) or (dataset==15):
        print("DNA Methylation Platform GPL16304, ", end="")
    if (dataset==7) or (dataset==10) or (dataset==11):
        print("Gene Expression HTSeq Count, ", end="")
    if (dataset==8) or (dataset==12) or (dataset==13):
        print("Gene Expression HTSeq FPKM, ", end="")
    if (dataset==9) or (dataset==14) or (dataset==15):
        print("Gene Expression HTSeq FPKM-UQ, ", end="")
    print("miRNA Expression (Theano)\n")

    # Dataset amount
    if (dataset>=7) and (dataset<=9):       # Gene + miRNA
        n_dataset = 2
    elif (dataset>=10) and (dataset<=15):   # Methylation + Gene + miRNA
        n_dataset = 3
    
    # Load datasets
    datasets = load_data(dataset, pca)

    #############################################################################
    ################################# DBN LVL-1 #################################
    #############################################################################
    # List of parameters
    Ws_0 = []   # params for low-lvl DBN DNA Methylation part
    bs_0 = []
    Ws_1 = []   # params for low-lvl DBN Gene Expression part
    bs_1 = []
    Ws_2 = []   # params for low-lvl DBN miRNA Expression part
    bs_2 = []

    # Iterate for number of datasets
    for nr_dataset in range(n_dataset):
        # Title and dataset type
        if n_dataset == 2:
            if nr_dataset == 0:
                dataset_type = 1
                print("\nDBN lvl 1: Gene Expression data\n")
            elif nr_dataset == 1:
                dataset_type = 2
                print("\nDBN lvl 1: miRNA Expression data\n")
        if n_dataset == 3:
            if nr_dataset == 0:
                dataset_type = 0
                print("\nDBN lvl 1: DNA Methylation data\n")
            elif nr_dataset == 1:
                dataset_type = 1
                print("\nDBN lvl 1: Gene Expression data\n")
            elif nr_dataset == 2:
                dataset_type = 2
                print("\nDBN lvl 1: miRNA Expression data\n")

        ############################## PREPARE DATASET ##############################
        # take input and label set
        if n_dataset == 2:
            input_set = datasets[nr_dataset*2]
            label_set = datasets[(nr_dataset*2)+1]
        elif n_dataset == 3:
            input_set = datasets[nr_dataset*2]
            label_set = datasets[(nr_dataset*2)+1]

        # Split dataset into training and test set
        train_input_set, test_input_set, train_label_set, test_label_set = train_test_split(input_set, label_set, test_size=0.25, random_state=100)
        # Size of input layer
        _, nr_in = train_input_set.shape
        # Number of training batches
        n_train_batches = train_input_set.shape[0] // batch_size

        # cast inputs and labels as shared variable to accelerate computation
        train_set_x, train_set_y = shared_dataset(data_xy = (train_input_set,train_label_set))


        ############################### BUILD NN MODEL ##############################
        print('Build NN Model')
        numpy_rng = numpy.random.RandomState(123)
        
        if dataset_type == 0:
            dbn = DBN(numpy_rng=numpy_rng, n_ins=nr_in, hidden_layers_sizes=layers_met, n_outs=layers_met[-1])
        elif dataset_type == 1:
            dbn = DBN(numpy_rng=numpy_rng, n_ins=nr_in, hidden_layers_sizes=layers_gen, n_outs=layers_gen[-1])
        elif dataset_type == 2:
            dbn = DBN(numpy_rng=numpy_rng, n_ins=nr_in, hidden_layers_sizes=layers_mir, n_outs=layers_mir[-1])


        ############################# PRETRAIN NN MODEL #############################
        print('Pretrain NN Model')

        # Get the pretraining functions. It is on the amount of the number of layers.
        pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size, k=k)

        # iterate for each RBMs
        for i in range(dbn.n_layers):
            # iterate for pretraining epochs
            for epoch in range(pretraining_epochs):
                c = []
                # iterate for number of training batches
                for batch_index in range(n_train_batches):
                    # c is a list of monitoring cost per batch for RBM[i]
                    c.append(pretraining_fns[i](index=batch_index, lr=pretrain_lr))

        
        ########################### SAVE DBN LVL-1 RESULTS ###########################
        # save Ws,bs from [W,b,W,b,...,W,b] of dbn.params
        if dataset_type == 0:
            for idx_param in range(len(layers_met)):
                Ws_0.extend([dbn.params[idx_param*2]])
                bs_0.extend([dbn.params[(idx_param*2)+1]])
        elif dataset_type == 1:
            for idx_param in range(len(layers_gen)):
                Ws_1.extend([dbn.params[idx_param*2]])
                bs_1.extend([dbn.params[(idx_param*2)+1]])
        elif dataset_type == 2:
            for idx_param in range(len(layers_mir)):
                Ws_2.extend([dbn.params[idx_param*2]])
                bs_2.extend([dbn.params[(idx_param*2)+1]])

        # save output (last layer output of DBN lvl-1)
        output_model = dbn.pretraining_bottom_layer_output(train_set_x=train_set_x, dropout=dropout)
        if dataset_type == 0:
            dbn_lvl1_out_0 = output_model()
        elif dataset_type == 1:
            dbn_lvl1_out_1 = output_model()
        elif dataset_type == 2:
            dbn_lvl1_out_2 = output_model()
    
    
    
    #############################################################################
    ################################# DBN LVL-2 #################################
    #############################################################################
    # Title for DBN lvl-2
    print("\nDBN lvl 2\n")


    ############################## PREPARE DATASET ##############################
    # Concatenate output of DBN lvl-1 as input for DBN lvl-2
    if n_dataset == 2:
        train_input_set = numpy.concatenate((dbn_lvl1_out_1,dbn_lvl1_out_2),axis=1)
    elif n_dataset == 3:
        train_input_set = numpy.concatenate((numpy.concatenate((dbn_lvl1_out_0,dbn_lvl1_out_1),axis=1),dbn_lvl1_out_2), axis=1)
    
    # Size of input layer
    _, nr_in = train_input_set.shape
    # Number of training batches
    n_train_batches = train_input_set.shape[0] // batch_size

    # cast inputs and labels as shared variable to accelerate computation
    train_set_x = theano.shared(numpy.asarray(train_input_set, dtype=theano.config.floatX), borrow=True)


    ############################### BUILD NN MODEL ##############################
    print('Build NN Model')
    numpy_rng = numpy.random.RandomState(123)
    dbn = DBN(numpy_rng=numpy_rng, n_ins=nr_in, hidden_layers_sizes=layers_tot, n_outs=1)


    ############################# PRETRAIN NN MODEL #############################
    print('Pretrain NN Model')
    
    # Get the pretraining functions. It is on the amount of the number of layers.
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size, k=k)

    # iterate for each RBMs
    for i in range(dbn.n_layers):
        # iterate for pretraining epochs
        for epoch in range(pretraining_epochs):
            c = []
            # iterate for number of training batches
            for batch_index in range(n_train_batches):
                # c is a list of monitoring cost per batch for RBM[i]
                c.append(pretraining_fns[i](index=batch_index, lr=pretrain_lr))


    ########################### SAVE DBN LVL-2 RESULTS ##########################
    Ws_3 = []   # params for lvl-2 DBN
    bs_3 = []

    for idx_param in range(len(layers_tot)):
        Ws_3.extend([dbn.params[idx_param*2]])
        bs_3.extend([dbn.params[(idx_param*2)+1]])


    
    #############################################################################
    #################################### MDBN ###################################
    #############################################################################
    # Title for mDBN
    print("\nmDBN\n")


    ############################## PREPARE DATASET ##############################
    # 1. input + output
    train_sets_x = []
    test_sets_x = []
    nrs_in = []

    for nr_dataset in range(n_dataset):
        # take input and label set
        if n_dataset == 2:
            input_set = datasets[nr_dataset*2]
            label_set = datasets[(nr_dataset*2)+1]
        elif n_dataset == 3:
            input_set = datasets[nr_dataset*2]
            label_set = datasets[(nr_dataset*2)+1]

        # Split dataset into training and test set
        train_input_set, test_input_set, train_label_set, test_label_set = train_test_split(input_set, label_set, test_size=0.25, random_state=100)
        # Size of input layer
        _, nr_in = train_input_set.shape
        # Number of training batches
        n_train_batches = train_input_set.shape[0] // batch_size

        # cast inputs and labels as shared variable to accelerate computation
        train_set_x, train_set_y = shared_dataset(data_xy = (train_input_set,train_label_set))
        test_set_x, test_set_y = shared_dataset(data_xy = (test_input_set,test_label_set))

        # save in list
        train_sets_x.append(train_set_x)
        test_sets_x.append(test_set_x)
        nrs_in.append(nr_in)

    # 2. weights + biases
    if n_dataset == 2:
        W_lvl1 = [Ws_1,Ws_2]
        b_lvl1 = [bs_1,bs_2]
    elif n_dataset == 3:
        W_lvl1 = [Ws_0,Ws_1,Ws_2]
        b_lvl1 = [bs_0,bs_1,bs_2]

    # 3. layers setting
    if n_dataset == 2:
        hidden_layers_lvl1_sizes = [layers_gen,layers_mir]
    elif n_dataset == 3:
        hidden_layers_lvl1_sizes = [layers_met,layers_gen,layers_mir]



    ############################### BUILD NN MODEL ##############################
    print('Build NN Model')
    numpy_rng = numpy.random.RandomState(123)

    mdbn = mDBN(numpy_rng=numpy_rng,
            n_ins=nrs_in,
            hidden_layers_lvl1_sizes=hidden_layers_lvl1_sizes,
            hidden_layers_lvl2_sizes=layers_tot,
            n_outs=1,
            W_lvl1=W_lvl1,
            b_lvl1=b_lvl1,
            W_lvl2=Ws_3,
            b_lvl2=bs_3)

    
    ############################# FINETUNE NN MODEL #############################
    print('Train NN Model')
    
    # Get the training functions.
    train_fn = mdbn.build_finetune_functions(train_sets_x=train_sets_x, train_set_y=train_set_y, batch_size=batch_size, learning_rate=finetune_lr, dropout=dropout, optimizer=optimizer)

    # iterate for training epochs
    for j in range(training_epochs):
        # iterate for number of training batches
        for minibatch_index in range(n_train_batches):
            train_fn(minibatch_index)

    
    
    ############################### TEST NN MODEL ###############################
    print('Test NN Model')
    
    # Get the test functions.
    test_model = mdbn.predict(test_sets_x=test_sets_x, dropout=dropout)

    # take the test result
    test_predicted_label_set = test_model()

    # accuracy, p, r, f, s
    mse = mean_squared_error(test_label_set, test_predicted_label_set)
    r2 = r2_score(test_label_set, test_predicted_label_set)

    # print results
    print("MSE = " + str(mse))
    print("R2 = " + str(r2))


if __name__ == '__main__':
    start = timeit.default_timer()

    print("\n\nWhat type of features do you want to use?")
    print("[1] DNA Methylation")
    print("[2] Gene Expression")
    print("[3] miRNA Expression")
    
    try:
        features = input("Insert here [default = 3]: ")
    except Exception as e:
        features = 3

    if features == 1:   # if DNA Methylation is picked
        print("You will use DNA Methylation data to create the prediction")
        print("\nWhat type DNA Methylation data do you want to use?")
        print("[1] Platform GPL8490\t(27578 cpg sites)")
        print("[2] Platform GPL16304\t(485577 cpg sites)")
        try:
            met = input("Insert here [default = 1]: ")
        except Exception as e:
            met = 1
        
        if met == 2: # if Platform GPL16304 is picked
            print("You will use DNA Methylation Platform GPL16304 data")
            DATASET = 2
        else:       # if Platform GPL8490 or any other number is picked
            print("You will use DNA Methylation Platform GPL8490 data")
            DATASET = 1
        
    elif features == 2: # if Gene Expression is picked
        print("You will use Gene Expression data to create the prediction")
        print("\nWhat type Gene Expression data do you want to use?")
        print("[1] Count")
        print("[2] FPKM")
        print("[3] FPKM-UQ")
        try:
            gen = input("Insert here [default = 1]: ")
        except Exception as e:
            gen = 1
        
        if gen == 2:    # if FPKM is picked
            print("You will use Gene Expression FPKM data")
            DATASET = 4
        elif gen == 3:  # if FPKM-UQ is picked
            print("You will use Gene Expression FPKM-UQ data")
            DATASET = 5
        else:           # if Count or any other number is picked
            print("You will use Gene Expression Count data")
            DATASET = 3
        
    else:   # if miRNA Expression or any other number is picked
        DATASET = 6
        print("You will use miRNA Expression data to create the prediction")

    test_mDBN(dataset=DATASET)

    stop = timeit.default_timer()
    print(stop-start)
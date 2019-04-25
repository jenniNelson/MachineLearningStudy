from sys import path
from os.path import dirname, realpath


MY_DIR = dirname(realpath(__file__))
path.append(MY_DIR)
PARENT_DIR = dirname(path[0])
path.append(PARENT_DIR)


# from numpy.core._multiarray_umath import ndarray
from numpy import genfromtxt
from math import exp
import numpy as np

from random import shuffle

def sigmoid(x):
    # Numerically stable sigmoid
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)


def get_and_append_data(relative_filepath):

    data = genfromtxt(MY_DIR + "/" + relative_filepath, delimiter=',')

    print(data.shape)
    # fill = np.ones( (data.shape[0],1) )
    # print(fill.shape)
    # data = np.concatenate( [data, fill] , axis=1)

    data = np.insert(data, -1, 1.0, axis=1)
    data[:, -1] -= .5
    data[:, -1] *= 2

    return data






class NurwalNetwurk:
    def __init__(self, hidden_layer_width, input_width,  initialization_scheme='ones', num_layers=4):
        """
        num_layers includes both the input and output layer.
        So a network with two hidden layers has 4 total layers, and 3 layers for weights
        """

        self.num_layers = num_layers
        self.input_width = input_width
        self.hidden_layer_width = hidden_layer_width
        self.initialization_scheme = initialization_scheme

        self.neuron_cache = []
        self.linearCombo_cache = []
        self.dL_dW_cache = []
        self.dL_dZ_cache = []
        self.weights = []

        self.clear_weights()
        self.clear_innards()

        #
        # print("weights")
        # for weightLayer in self.weights:
        #     print(weightLayer.shape)
        # print("dL_dW_cache")
        # for dWLayer in self.dL_dW_cache:
        #     print(dWLayer.shape)
        # print("dL_dZ_cache")
        # for dZ in self.dL_dZ_cache:
        #     print(dZ.shape)
        # print("neuron_cache")
        # for neuronLayer in self.neuron_cache:
        #     print(neuronLayer.shape)
        # print("linearCombo_cache")
        # for linCombLayer in self.linearCombo_cache:
        #     print(linCombLayer.shape)

    # Initialize weights.
    # self.weights is a list of np arrays of the connections between layers
    # each array is shape (to_layer_width, from_layer_width)
    # the weight from layer i node #from to layer i+1 node #to is at weights[i][to][from]
    # This doesn't connect to the "bias" node, which is always 1
    # and is last in any list of hidden layer nodes
    def clear_weights(self):
        """ Initialize weights.
        self.weights is a list of np arrays of the connections between layers
        each array is shape (to_layer_width, from_layer_width)
        the weight from layer i node #from to layer i+1 node #to is at weights[i][to][from]
        This doesn't connect to the "bias" node, which is always 1
        and is last in any list of hidden layer nodes
        """
        # 0th is input to a hidden layer (except bias node) from the input
        self.weights = [self.init_w_scheme(self.hidden_layer_width - 1, self.input_width)]

        # Middle connections: num_layers-1 - input - output
        for hidden_layer in range(self.num_layers-3):
            self.weights.append(self.init_w_scheme(self.hidden_layer_width - 1, self.hidden_layer_width))

        # Last connects a hidden layer with the output
        self.weights.append(self.init_w_scheme(1, self.hidden_layer_width) ) # that last one should



    def clear_innards(self):
        self.neuron_cache = [ np.full(self.input_width, np.nan)]
        self.linearCombo_cache = [ np.full(self.input_width, np.nan)]
        self.dL_dZ_cache = [np.full(self.input_width, np.nan)]
        self.dL_dW_cache = [ np.full( (self.hidden_layer_width - 1, self.input_width), np.nan)]

        for hidden_layer in range(self.num_layers-2):
            self.neuron_cache.append(np.full( self.hidden_layer_width, np.nan))
            self.linearCombo_cache.append(np.full(self.hidden_layer_width, np.nan))
            self.dL_dZ_cache.append(np.full(self.hidden_layer_width, np.nan))
            self.dL_dW_cache.append(np.full( (self.hidden_layer_width -1, self.hidden_layer_width),  np.nan) )

        self.neuron_cache.append(np.full(1, np.nan))
        self.linearCombo_cache.append(np.full((1, 1), np.nan))
        self.dL_dZ_cache.append(np.full((1, 1), np.nan))
        self.dL_dW_cache[-1] = np.full( (1, self.hidden_layer_width), np.nan)



    def init_w_scheme(self, d1, d2):
        if self.initialization_scheme == 'zeros':
            return  np.zeros((d1, d2))
        elif self.initialization_scheme == 'random':
            return np.random.normal(0,1, (d1, d2))
        elif self.initialization_scheme == 'ones':
            return  np.ones((d1, d2))

    # def pass_forward(self, input):
    #     if len(input) != self.input_width:
    #         print("Input vector to pass_forward incorrect should be {} but is {}".format(self.input_width, len(input)))
    #
    #     layers = [input, np.zeros(self.hidden_layer_width), np.zeros(self.hidden_layer_width)]
    #
    #     # Propogate through hidden layers
    #     for layer in range(0, self.num_layers-1):
    #         for node_idx in range(0, self.hidden_layer_width):
    #             linear_combo = np.dot(layers[layer], self.weights[layer][node_idx])
    #             layers[layer][node_idx] = sigmoid(linear_combo)
    #         layers[layer][-1] = 1




    def backpropagate(self, sample):

        if len(sample)-1 != self.input_width:
            print("Sample not proper input width")

        self.input_value(sample[:-1])

        for layer in range(self.num_layers-2, -1, -1): # backwards in layers (num_layers one big to include y)

            for toNodeIdx in range(0, self.dL_dW_cache[layer].shape[0]):

                for fromNodeIdx in range(0, self.dL_dW_cache[layer].shape[1]):
                    self.dL_dW(layer, fromNodeIdx, toNodeIdx, sample[-1])

        return self.dL_dW_cache


    def train(self, training_set, epochs, learning_rate_iterator):

        for epoch in range(epochs):
            np.random.shuffle(training_set)
            next_rate = next(learning_rate_iterator)
            for sample in training_set:
                # print("NEXT SAMPLE: ", sample)
                weight_gradient = self.backpropagate(sample)
                for layer in range(len(self.weights)):
                    self.weights[layer] -= next_rate * weight_gradient[layer]

    def predict(self, input):

        self.input_value(input)

        for layer in range(0, self.num_layers): # backwards in layers (num_layers one big to include y)
            for nodeIdx in range(0, self.layer_width(layer)):
                self.neuronVal(layer, nodeIdx);

        return self.neuronVal(self.num_layers-1,0)




    def test(self, input):

        self.input_value(input[:-1])
        result = self.neuronVal(self.num_layers-1, 0) # y node
        # result = self.pass_forward(input[:-1])
        # result = self.predict(input[:-1]);
        # print("prediction: ", result, " actual: ", input[-1])
        signs_match = input[-1] * result

        return signs_match > 0

    def error_rate(self, testing_data):
        num_incorrect = 0
        total = len(testing_data)
        for sample in testing_data:
            if not self.test(sample) :
                num_incorrect += 1
        return num_incorrect / total


    def precision_and_recall(self, testing_data):
        num_pos_in_data = 0
        num_correct_of_pos_labels = 0
        num_neg_in_data = 0
        num_correct_of_neg_labels = 0

        for sample in testing_data:
            result = self.test(sample);
            if sample[-1] < 0:
                num_neg_in_data +=1
                if result:
                    num_correct_of_neg_labels += 1
            if sample[-1] > 0:
                num_pos_in_data += 1
                if result:
                    num_correct_of_pos_labels += 1
        return (num_correct_of_neg_labels + num_correct_of_pos_labels)/(len(training_data)) \
            , num_correct_of_pos_labels/num_pos_in_data \
            , num_correct_of_neg_labels / num_neg_in_data


    def input_value(self, input):
        if len(input) != self.input_width:
            print("Input width invalid.")
        self.clear_innards()
        self.neuron_cache[0] = input

    def linearCombo(self, layer, node):

        if not np.isnan(self.linearCombo_cache[layer][node]):
            # print("In cache!")
            return self.linearCombo_cache[layer][node]

        # Bias node?
        if node == self.layer_width(layer) -1 and layer != self.num_layers-1:
            self.linearCombo_cache[layer][node] = 1
            return 1

        # if neuron layer below bad, make it not bad
        # (assume weights is always good)
        for lower_node in range(0, self.layer_width(layer-1) ):
            if np.isnan(self.neuron_cache[layer-1][lower_node]):
                # print("neuron [{}][{}] not in cache".format(layer-1, lower_node))
                self.neuronVal(layer-1, lower_node)

        value = np.dot(self.neuron_cache[layer-1], self.weights[layer-1][node])
        # print(value)
        self.linearCombo_cache[layer][node] = value
        return value

    def neuronVal(self, layer, node):

        if layer == self.num_layers-1 and node != 0:
            print("Why you calling the last layer with something other than node 0?\n There's only one node--y")

        if not np.isnan(self.neuron_cache[layer][node]):
            return self.neuron_cache[layer][node]

        if layer == self.num_layers-1 and node == 0: # this neuron is y:
            value = self.linearCombo(layer, node)
            self.neuron_cache[layer][node] = value
            return value

        if node == self.layer_width(layer) -1: # if bias node:
            self.neuron_cache[layer][node] = 1
            return 1


        value = sigmoid(self.linearCombo(layer, node))


        self.neuron_cache[layer][node] = value
        return value

    def dZ_dZ(self, topZlayer, topZnode, bottomZnode):

        # if bottomZnode == self.layer_width(topZlayer-1):
        #     return 1
        return self.dSigma_dS(topZlayer, topZnode) * self.dS_dZ(topZlayer, topZnode, bottomZnode)
        pass

    def dZ_dW(self, topZlayer, topZnode, bottomWfrom):
        if topZlayer == self.num_layers-1:
            return self.dS_dW(topZlayer, bottomWfrom)

        return self.dSigma_dS(topZlayer, topZnode) * self.dS_dW(topZlayer, bottomWfrom)

    def dS_dZ(self, topSlayer, topSnode, bottomZnode):
        val = self.weights[topSlayer-1][topSnode][bottomZnode]
        return val

    def dS_dW(self, topSlayer, bottomWfrom):
        """ds[layer][node] / dw[layer-1][from][(to) node] = z[layer-1][from]"""
        return self.neuronVal(topSlayer-1, bottomWfrom)

    def dSigma_dS(self, sLayer, sNode):
        sig = sigmoid( self.linearCombo(sLayer, sNode) )
        return sig * (1-sig)

    def dL_dZ(self, zLayer, zNode, true_label):

        # If cache exist, use that
        if not np.isnan(self.dL_dZ_cache[zLayer][zNode]):
            return self.dL_dZ_cache[zLayer][zNode]

        # If this is the y neuron, treat it special
        if zLayer == self.num_layers-1: #TODO Check off by one error here
            value = self.dL_dy(true_label)
            self.dL_dZ_cache[zLayer][zNode] = value
            return value

        # If this is a bias neuron (constant 1):
        if zNode == self.layer_width(zLayer) - 1 - 1: #TODO Check off by one error here
            self.dL_dZ_cache[zLayer][zNode] = 0
            return 0

        # If this is in the last layer above the y neuron, it doesn't use sigma
        if zLayer == self.num_layers -2: #TODO Check off by one error here
            value = self.dL_dy(true_label) * self.dS_dZ(zLayer+1, 0, zNode)
            self.dL_dZ_cache[zLayer][zNode] = value
            return value


        above_layer_width = self.layer_width(zLayer+1)
        sum = 0
        for i in range(0, above_layer_width-1):
            sum += self.dL_dZ(zLayer + 1, i, true_label) * self.dZ_dZ(zLayer+1, i, zNode)

        # Cache this one
        self.dL_dZ_cache[zLayer][zNode] = sum

        return sum

    def layer_width(self, layer):
        return len(self.neuron_cache[layer])

    def dL_dW(self, wLayer, wFrom, wTo, true_label):

        if not np.isnan(self.dL_dW_cache[wLayer][wTo][wFrom]):
            return self.dL_dW_cache[wLayer][wTo][wFrom]

        # TODO Check this okay???
        if wLayer == self.num_layers-2: # last weight layer:
            value = self.dL_dy(true_label) * self.dS_dW(wLayer +1, wFrom)
            self.dL_dW_cache[wLayer][wTo][wFrom] = value
            return value

        result = self.dL_dZ(wLayer+1, wTo, true_label) * self.dZ_dW(wLayer+1,wTo,wFrom)

        self.dL_dW_cache[wLayer][wTo][wFrom] = result

        return result

    def dL_dy(self, true_label):
        return self.neuronVal(self.num_layers-1, 0) - true_label






def learning_rate1(initial, d):
    yield initial
    iter = 0
    while True:
        yield initial / (1 + (initial/d)*iter)
        iter += 1

def learning_rate_constant(c):
    while True:
        yield c



if __name__ == '__main__':

    bank_train_file = "Data/bank-note/train.csv"
    bank_test_file = "Data/bank-note/test.csv"

    training_data = get_and_append_data(bank_train_file)
    testing_data = get_and_append_data(bank_test_file)

    # print(testing_data[:], "...")
    # print(training_data)


    # homework_network = NurwalNetwurk( hidden_layer_width=3, input_width=3, initialization_scheme='ones', num_layers=4)
    #
    # initial_weights = [  ]
    #
    # gradient = homework_network.backpropagate([])



    # network = NurwalNetwurk( hidden_layer_width=10, input_width=training_data.shape[1]-1 , initialization_scheme='zeros', num_layers=4)
    #
    # network.train(training_data, 10, learning_rate1(1, 1))
    #
    # train_err = network.error_rate(training_data[:])
    # print("train err: ", train_err)
    # test_err = network.error_rate(testing_data[:])
    # print("test err: ", test_err)

    # print(training_data)

    # train_precision, train_pos_recall, train_neg_recall = network.precision_and_recall(training_data)
    # test_precision, test_pos_recall, test_neg_recall = network.precision_and_recall(testing_data)
    #
    # print( "Training: prec: {}, pos_recall: {}, neg_recall: {} ".format(train_precision, train_pos_recall, train_neg_recall))
    # print("Testing: prec: {}, pos_recall: {}, neg_recall: {} ".format(test_precision, test_pos_recall, test_neg_recall))

    print_form = "{:<7.0f}\t&{:>10.15f}\t&{:>10.15f}\t\\\\\hline"

    for init_scheme in ["random", "zeros"]:
        print("SCHEME: ", init_scheme, "\n")
        for width in [5, 10, 25, 50, 100]:


            network = NurwalNetwurk(hidden_layer_width=width, input_width=training_data.shape[1] - 1,
                                    initialization_scheme=init_scheme, num_layers=4)

            network.train(training_data, 15, learning_rate1(.1, .01))

            train_err = network.error_rate(training_data[:])
            # print("train err: ", train_err)
            test_err = network.error_rate(testing_data[:])
            # print("test err: ", test_err)
            print(print_form.format(width, train_err, test_err))



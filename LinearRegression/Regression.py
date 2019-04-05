from sys import path
from os.path import dirname, realpath
MY_DIR = dirname(realpath(__file__))
path.append(MY_DIR)
PARENT_DIR = dirname(path[0])
path.append(PARENT_DIR)

import numpy

from random import shuffle


class Example:
    def __init__(self, values, weight=1):
        self.attributes = values[0: len(values)-1]
        self.attributes.append(1)
        self.label = values[len(values)-1]
        self.weight = weight


def examples_from_file_with_b(filename):
    examples = list()
    with open(filename, 'r') as train_data:
        for line in train_data:
            terms = line.strip().split(',')
            for idx in range(len(terms)):
                try:
                    terms[idx] = float(terms[idx])
                except ValueError:
                    print("Bad data! No! Stop it!")
            examples.append(Example(terms))
    return examples


def lms_gradient_descent(examples, learning_rate, threshold, max_iterations):
    num_attributes = len(examples[0].attributes)
    weights = [0] * num_attributes  # PYTHON WOOOOOoOoOooo
    error = lms_error(examples, weights)
    weight_diff = threshold + 1
    iteration = 0
    while error > threshold and iteration != max_iterations and weight_diff > 1e-6:
        # print(iteration, ",", error)
        gradient = compute_gradient(examples, weights)
        for i in range(0,num_attributes):
            weights[i] -= learning_rate * gradient[i]
        error = lms_error(examples, weights)
        iteration += 1
    return weights, iteration, error


def predict(sample, weights):
    num_attributes = len(sample.attributes)
    total = 0
    # diff = sample.label
    for i in range(0, num_attributes):
        total += weights[i] * sample.attributes[i]
    return total

def stochastic_descent(examples, learning_rate, threshold, max_iterations):
    num_attributes = len(examples[0].attributes)
    weights = [0] * num_attributes  # PYTHON WOOOOOoOoOooo
    # old_weights = copy.deepcopy(weights)
    # error = lms_error(examples, weights)
    count_updates = 0
    for iteration in range(0, max_iterations):
        shuffle(examples)
        for sample in examples:
            stochastic_gradient = learning_rate * (sample.label - sum(i*j for i,j in zip(weights, sample.attributes)))
            for atr in range(0, num_attributes):
                weights[atr] += stochastic_gradient * sample.attributes[atr]
            error = lms_error(examples, weights)
            # print(count_updates, ",", error)
            count_updates += 1
            if error < threshold:
                return weights, iteration, error
    return weights, iteration, error

def grad_and_err(examples, weights):
    num_attributes = len(examples[0].attributes)
    gradient = [0] * num_attributes
    err_total = 0
    for sample in examples:
        diff = sample.label
        for i in range(0, num_attributes):
            diff -= weights[i] * sample.attributes[i]
        for which_atr in range(0, num_attributes):
            gradient[which_atr] += -diff * sample.attributes[which_atr]
        err_total += diff * diff
    return .5 * err_total, gradient

def lms_error(examples, weights):
    num_attributes = len(examples[0].attributes)
    total = 0
    for sample in examples:
        diff = sample.label
        for i in range(0, num_attributes):
            diff -= weights[i] * sample.attributes[i]
        total += diff*diff
    return .5*total

def compute_gradient(examples, weights):
    gradient = list()
    num_attributes = len(examples[0].attributes)
    for which_weight in range(0, num_attributes):
        total = 0
        for sample in examples:
            diff = sample.label
            for atr in range(0, num_attributes):
                diff -= weights[atr] * sample.attributes[atr]
            total += diff * sample.attributes[which_weight]
        gradient.append(-total)
    return gradient


if __name__ == '__main__':
    concrete_train_file = MY_DIR + "/Data/concrete/train.csv"
    concrete_test_file = MY_DIR + "/Data/concrete/test.csv"

    concrete_train_examples = examples_from_file_with_b(concrete_train_file)
    concrete_test_examples = examples_from_file_with_b(concrete_test_file)

    format_string = "{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t\\\\\hline"

    # print("Batch gradient Descent: r=0.015, 1000 Epochs")
    print("r		&	$w_1$	&	$w_2$	&	$w_3$	&	$w_4$	&	$w_5$	&	$w_6$	&	$w_7$	& 	$b$	&	Error  \\\hline\hline")
    solution, attempts, error = lms_gradient_descent(concrete_train_examples, 0.015, 0.01, 1000)
    print(format_string.format(0.015,solution[0],solution[1],solution[2],solution[3],solution[4],solution[5],solution[6],solution[7],error))

    # print("Batch gradient Descent: r=0.01, 1000 Epochs")
    solution, attempts, error = lms_gradient_descent(concrete_train_examples, 0.01, 0.01, 1000)
    print(format_string.format(0.01,solution[0],solution[1],solution[2],solution[3],solution[4],solution[5],solution[6],solution[7],error))


    # print("Batch gradient Descent: r=0.001, 1000 Epochs")
    solution, attempts, error = lms_gradient_descent(concrete_train_examples, 0.001, 0.01, 1000)
    print(format_string.format(0.001,solution[0],solution[1],solution[2],solution[3],solution[4],solution[5],solution[6],solution[7],error))

    # print("Batch gradient Descent: r=0.0001, 1000 Epochs")
    solution, attempts, error = lms_gradient_descent(concrete_train_examples, 0.0001, 0.01, 1000)
    print(format_string.format(0.0001,solution[0],solution[1],solution[2],solution[3],solution[4],solution[5],solution[6],solution[7],error))

    print("\n")

    print("r		&	$w_1$	&	$w_2$	&	$w_3$	&	$w_4$	&	$w_5$	&	$w_6$	&	$w_7$	& 	$b$	&	Error  \\\hline\hline")
    # print("Stochastic gradient Descent: r=0.015, 1000 Epochs")
    solution, attempts, error = stochastic_descent(concrete_train_examples, 0.015, 0.01, 100)
    print(format_string.format(0.015,solution[0],solution[1],solution[2],solution[3],solution[4],solution[5],solution[6],solution[7],error))
    # print("\n")

    # print("Stochastic gradient Descent: r=0.01, 1000 Epochs")
    solution, attempts, error = stochastic_descent(concrete_train_examples, 0.01, 0.01, 100)
    print(format_string.format(0.01,solution[0],solution[1],solution[2],solution[3],solution[4],solution[5],solution[6],solution[7],error))
    # print("\n")

    # print("Stochastic gradient Descent: r=0.001, 1000 Epochs")
    solution, attempts, error = stochastic_descent(concrete_train_examples, 0.001, 0.01, 100)
    print(format_string.format(0.001,solution[0],solution[1],solution[2],solution[3],solution[4],solution[5],solution[6],solution[7],error))
    # print("\n")

    # print("Stochastic gradient Descent: r=0.0001, 1000 Epochs")
    solution, attempts, error = stochastic_descent(concrete_train_examples, 0.0001, 0.01, 100)
    print(format_string.format(0.0001,solution[0],solution[1],solution[2],solution[3],solution[4],solution[5],solution[6],solution[7],error))


    print("\nAnalytically Derived Weight")


    # Slapped on some numpy jazz in a panic at the end. Apologies for the mess.

    # ((float)(thing) for thing in list)
    labellist = list(sample.label for sample in concrete_train_examples)
    # print(labellist)
    labels = numpy.matrix(labellist)
    # print(labels)
    # labels = numpy.array((float)(sample.label) for sample in concrete_train_examples)
    attributes = numpy.matrix(list(sample.attributes for sample in concrete_train_examples), numpy.float_)
    XY = numpy.matmul(attributes.transpose(), labels.transpose())

    X_transposed = numpy.transpose(attributes)

    XXt = numpy.matmul(attributes.transpose(), X_transposed.transpose())
    # print(XXt)
    XXt_inv = numpy.linalg.inv(XXt)

    weights = numpy.matmul(XXt_inv, XY)

    print(weights)





from sys import path
from os.path import dirname, realpath
MY_DIR = dirname(realpath(__file__))
path.append(MY_DIR)
PARENT_DIR = dirname(path[0])
path.append(PARENT_DIR)


import numpy
from random import shuffle

from copy import deepcopy

class Example:
    def __init__(self, values, weight=1):
        self.attributes = numpy.array(values[0: len(values)-1])
        self.attributes = numpy.append(self.attributes, [1])
        # self.attributes = values[0: len(values)-1]
        # self.attributes.append(1)
        self.label = -1 if values[-1] == 0 else 1
        # self.label = values[len(values)-1]
        self.weight = weight


def examples_from_file(filename):
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


def predict_standard(sample, weights):
    predict_val = numpy.dot( weights, sample.attributes)
    return -1 if predict_val < 0 else 1
    # return predict_val

def evaluate_weights(sample, weights):
    predict_val = sample.label * numpy.dot( weights, sample.attributes)
    return predict_val < 1

def primal_svm(examples, epochs, learning_rate, C):
    weights = numpy.zeros(len(examples[0].attributes))
    for _ in range(epochs):
        shuffle(examples)
        for sample in examples:
            rate = next(learning_rate)
            if evaluate_weights(sample, weights):
                weights = (1-rate)* numpy.append(weights[:-1], [0]) + C*rate*len(examples)*sample.label*sample.attributes;
            else:
                weights[:-1] *= (1-rate);
    return weights


def average_error(examples, classifier):
    incorrect = 0
    correct = 0
    total = 0
    for sample in examples:
        if classifier(sample) * sample.label <= 0:
            incorrect += 1  # sample.weight
        else:
            correct += 1  #sample.weight
        total += 1
    return incorrect / total


if __name__ == '__main__':
    banknote_train_file = MY_DIR + "/Data/bank-note/train.csv"
    banknote_test_file = MY_DIR + "/Data/bank-note/test.csv"

    banknote_train_examples = examples_from_file(banknote_train_file)
    banknote_test_examples = examples_from_file(banknote_test_file)

    def learning_rate1(initial, d):
        yield initial
        iter = 0
        while True:
            yield initial / (1 + (initial/d)*iter)
            iter += 1

    def learning_rate2(initial):
        yield initial
        iter = 0
        while True:
            yield initial / (1 + iter)
            iter += 1



    epochs = 100
    C = 1


    format_string = "{:<7.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t\\\\\hline"


    # TUNING the learning rate for first learning rate equation. init = .1 d=1 is great
    # for init in numpy.linspace(0.1,1,9):
    #     for d in numpy.linspace(0.1,1,9):
    #         std_weights = primal_svm(banknote_train_examples, epochs, learning_rate1(init, d), C)
    #         avg_error_std_test = average_error(banknote_test_examples, lambda sample: predict_standard(sample, std_weights))
    #         avg_error_std_train = average_error(banknote_train_examples, lambda sample: predict_standard(sample, std_weights))
    #         print(init, ": ", d)
    #         # print(std_weights)
    #         print(avg_error_std_test, avg_error_std_train)
    #         # print(avg_error_std_train)
    #         print()

    # TUNING the learning rate for second learning rate equation. init = .4 is great
    # for init in numpy.linspace(0.1,1,15):
    #     std_weights = primal_svm(banknote_train_examples, epochs, learning_rate2(init), C)
    #     avg_error_std_test = average_error(banknote_test_examples, lambda sample: predict_standard(sample, std_weights))
    #     avg_error_std_train = average_error(banknote_train_examples, lambda sample: predict_standard(sample, std_weights))
    #     print(init)
    #     # print(std_weights)
    #     print(avg_error_std_test, avg_error_std_train)
    #     # print(avg_error_std_train)
    #     print()


    C_list = numpy.array([1, 10, 50, 100, 300, 500, 700]) / 873
    init = .1
    d = 1

    for C in C_list:
        weights = primal_svm(banknote_train_examples, epochs, learning_rate1(init, d), C)
        test_err = average_error(banknote_test_examples, lambda sample: predict_standard(sample, weights))
        train_err = average_error(banknote_train_examples, lambda sample: predict_standard(sample, weights))

        format()


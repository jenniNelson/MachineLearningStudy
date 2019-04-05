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

def predict_voted(sample, weight_list, num_correct_list):
    ultimate_decision = 0
    for i in range(len(weight_list)):
        decision = -1 if numpy.dot( weight_list[i], sample.attributes) < 0 else 1
        ultimate_decision += num_correct_list[i] * decision
    return -1 if ultimate_decision < 0 else 1

# def predict_averaged(sample, weight_list, num_correct_list):
#     ultimate_decision = 0
#     for i in range(len(weight_list)):
#         decision = numpy.multiply( numpy.transpose(weight_list[i]), sample.attributes)
#         ultimate_decision += num_correct_list[i] * decision
#     return -1 if ultimate_decision < 0 else 1

def standard_perceptron(examples, epochs, learning_rate):
    weights = numpy.zeros(len(examples[0].attributes))
    for _ in range(epochs):
        shuffle(examples)
        for sample in examples:
            if predict_standard(sample, weights) != sample.label:
                weights = weights +  learning_rate * sample.attributes * sample.label
    return weights

def voted_perceptron(examples, epochs, learning_rate):
    weights = numpy.zeros(len(examples[0].attributes))
    weight_list = list()
    correct_list = list()
    num_correct = 0
    for _ in range(epochs):
        shuffle(examples)
        for sample in examples:
            if predict_standard(sample, weights) != sample.label:
                weight_list.append(deepcopy(weights))
                correct_list.append(num_correct)
                weights += learning_rate * sample.attributes * sample.label
                num_correct = 1
            else:
                num_correct += 1
    return weight_list, correct_list

def average_perceptron(examples, epochs, learning_rate):
    weights = numpy.zeros(len(examples[0].attributes))
    average = numpy.zeros(len(examples[0].attributes))
    num_correct = 0
    for _ in range(epochs):
        shuffle(examples)
        for sample in examples:
            if predict_standard(sample, weights) != sample.label:
                weights += learning_rate * sample.attributes * sample.label
            else:
                average += weights
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

    learning_rate = 0.8
    epochs = 50


    format_string = "{:<7.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t&{:>10.3f}\t\\\\\hline"

    # print("Learning Rate:\t&Standard Method Test Error\t&Standard Method Train Error\t&Voting Method Test Error\t&Voting Method Train Error\t&Average Method Test Error\t&Average Method Train Error\t")

print("&\multicolumn{2}{c}{Standard}&\multicolumn{2}{c}{Voting}&\multicolumn{2}{c}{Averaging} \\ \hline")
print("Learning Rate:	&Test	&Train	&Test	&Train	&Test	&Train	\\\hline\hline")
    # print("Lr:")
    # print(learning_rate)

    std_weights = standard_perceptron(banknote_train_examples, epochs, learning_rate)

    avg_error_std_test = average_error(banknote_test_examples, lambda sample: predict_standard(sample, std_weights))
    avg_error_std_train = average_error(banknote_train_examples, lambda sample: predict_standard(sample, std_weights))

    # print(avg_error_std_test)
    # print(avg_error_std_train)
    # print(std_weights)

    # print()

    voted_weights, votes = voted_perceptron(banknote_train_examples, epochs, learning_rate)
    avg_error_voted_test = average_error(banknote_test_examples, lambda sample: predict_voted(sample, voted_weights, votes))
    avg_error_voted_train = average_error(banknote_train_examples, lambda sample: predict_voted(sample, voted_weights, votes))

    # print(avg_error_voted_test)
    # print(avg_error_voted_train)
    # # print(voted_weights)
    #
    # print()

    avg_weight = average_perceptron(banknote_train_examples, epochs, learning_rate)
    avg_error_avg_test = average_error(banknote_test_examples, lambda sample : predict_standard(sample, avg_weight))
    avg_error_avg_train = average_error(banknote_train_examples, lambda sample : predict_standard(sample, avg_weight))

    # print(avg_error_avg_test)
    # print(avg_error_avg_train)
    # print(avg_weight)


    print(format_string.format(learning_rate, avg_error_std_test, avg_error_std_train, avg_error_voted_test, avg_error_voted_train, avg_error_avg_test, avg_error_avg_train))
    print()
    print("Standard Method Weights: ")
    print(std_weights)
    print()
    print("Average Method Weights: ")
    print(avg_weight)
    print()
    print("Voting Weights and Counts: Total is ", len(voted_weights))
    weight_num = 0
    for weight_num in range(0, len(voted_weights), 1) :
        print(weight_num, "\t& ", voted_weights[weight_num], "\t& ", votes[weight_num])


    # Standard perceptron. With epoch = 10, report learned vector and avg error on test set
    # Voted perceptron. With epoch = 10, report all the learned weights and how many they got right, and avg error on test set
    # Average perceptron. Report learned weights. Compare to voted perceptrons. Report avg test error

    # Compare average errors for all three

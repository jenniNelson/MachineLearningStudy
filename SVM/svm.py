from sys import path
from os.path import dirname, realpath
MY_DIR = dirname(realpath(__file__))
path.append(MY_DIR)
PARENT_DIR = dirname(path[0])
path.append(PARENT_DIR)


import numpy as np
from random import shuffle
from scipy.optimize import minimize
from scipy.optimize import Bounds
from math import exp

class Example:
    def __init__(self, values, weight=1):
        self.attributes = np.array(values[0: len(values)-1])
        self.attributes = np.append(self.attributes, [1])
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
    predict_val = np.dot( weights, sample.attributes)
    return -1 if predict_val < 0 else 1
    # return predict_val

def evaluate_weights(sample, weights):
    predict_val = sample.label * np.dot( weights, sample.attributes)
    return predict_val < 1

def primal_svm(examples, epochs, learning_rate, C):
    weights = np.zeros(len(examples[0].attributes))
    for _ in range(epochs):
        shuffle(examples)
        for sample in examples:
            rate = next(learning_rate)
            if evaluate_weights(sample, weights):
                weights = (1-rate)* np.append(weights[:-1], [0]) + C*rate*len(examples)*sample.label*sample.attributes;
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





def dual_svm(x, y, C, kernel=np.dot):
    alpha_length = len(x)
    alpha_bounds = Bounds(np.full((alpha_length), 0),np.full((alpha_length), C))
    eq_constraints = {'type': 'eq',
                      'fun': lambda a: kernel(a, y)
                      }


    def loss(a):
        result = 0
        for i in range(alpha_length):
            internal_sum = 0
            for j in range(alpha_length):
                internal_sum += a[j]*y[j] * kernel(x[i], x[j])
            result += internal_sum * a[i] * y[i]
        return result -sum(a)


    init_guess = np.zeros(alpha_length)



    alphas = minimize(loss, init_guess, method="SLSQP", jac=False, constraints=[eq_constraints], bounds=alpha_bounds )
    # print(alphas)
    return alphas.x

def recover_w_from_alphas(a, x, y):
    w = np.zeros(len(x[0]))
    for i in range(len(a)):
        w += x[i] * a[i]*y[i]
    return w

def recover_b_from_alphas(a, w, x, y, threshold):
    b = 0
    num_nonzero = 0
    for i in range(len(a)):
        if a[i] > threshold:
            b+= y[i] - np.dot(w, x[i])
            num_nonzero += 1
    return b/num_nonzero

def recover_b_from_alphas_only(a, x, y, threshold):
    b = 0
    num_nonzero = 0
    for i in range(len(a)):
        if a[i] > threshold:
            inner_sum = 0
            for j in range(len(a)):
                inner_sum += a[j] * y[j] * np.dot(x[j], x[i])
            b+= y[i] - inner_sum
            num_nonzero += 1
    return b/num_nonzero

def recover_weights(a, x, y, threshold):
    w = recover_w_from_alphas(a, x, y)
    b = recover_b_from_alphas(a, w, x, y, threshold)
    # b = recover_b_from_alphas_only(a, x, y, threshold)
    return np.append(w, b)

def kernel_predict(a, x, y, kernel, sample):
    wdotx = 0
    for i in range(len(a)):
        wdotx += a[i]*y[i]*kernel(x[i], sample)

    b = 0
    num_b = 0
    for i in range(len(a)):
        inner_sum = 0
        for j in range(len(a)):
            inner_sum += a[j] * y[j] * kernel(x[i], x[j])
            num_b += 1
        b += y[i] - inner_sum

    result = wdotx + b/num_b

    return -1 if result < 0 else 1

def gaussian_kernel(w, x, gamma):
    return exp(- np.dot(w-x, w-x)/gamma)


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


    # TUNING the learning rate for first learning rate equation. init = .1 d=1 is great
    # for init in np.linspace(0.1,1,9):
    #     for d in np.linspace(0.1,1,9):
    #         std_weights = primal_svm(banknote_train_examples, epochs, learning_rate1(init, d), C)
    #         avg_error_std_test = average_error(banknote_test_examples, lambda sample: predict_standard(sample, std_weights))
    #         avg_error_std_train = average_error(banknote_train_examples, lambda sample: predict_standard(sample, std_weights))
    #         print(init, ": ", d)
    #         # print(std_weights)
    #         print(avg_error_std_test, avg_error_std_train)
    #         # print(avg_error_std_train)
    #         print()

    # TUNING the learning rate for second learning rate equation. init = .4 is great
    # for init in np.linspace(0.1,1,15):
    #     std_weights = primal_svm(banknote_train_examples, epochs, learning_rate2(init), C)
    #     avg_error_std_test = average_error(banknote_test_examples, lambda sample: predict_standard(sample, std_weights))
    #     avg_error_std_train = average_error(banknote_train_examples, lambda sample: predict_standard(sample, std_weights))
    #     print(init)
    #     # print(std_weights)
    #     print(avg_error_std_test, avg_error_std_train)
    #     # print(avg_error_std_train)
    #     print()


    C_list = np.array([1, 10, 50, 100, 300, 500, 700]) / 873
    init1 = .1
    d = 1

    format_string = "{:<7.3f}\t&{:>10.3f}\t&{:>10.3f}\t\\\\\hline"

    # for C in C_list:
    #     test_err = 0
    #     train_err = 0
    #     for _ in range(50):
    #         weights = primal_svm(banknote_train_examples, epochs, learning_rate1(init1, d), C)
    #         test_err += average_error(banknote_test_examples, lambda sample: predict_standard(sample, weights))
    #         train_err += average_error(banknote_train_examples, lambda sample: predict_standard(sample, weights))
    #
    #     print(format_string.format(C, train_err/50, test_err/50))
    #
    # print()
    #
    init2 = .4
    # for C in C_list:
    #     test_err = 0
    #     train_err = 0
    #     for _ in range(50):
    #         weights = primal_svm(banknote_train_examples, epochs, learning_rate2(init2), C)
    #         test_err += average_error(banknote_test_examples, lambda sample: predict_standard(sample, weights))
    #         train_err += average_error(banknote_train_examples, lambda sample: predict_standard(sample, weights))
    #
    #     print(format_string.format(C, train_err/50, test_err/50))

    # 
    # for C in C_list:
    #     weights_1 = primal_svm(banknote_train_examples, epochs, learning_rate1(init1, d), C)
    #     test_err_1 = average_error(banknote_test_examples, lambda sample: predict_standard(sample, weights_1))
    #     train_err_1 = average_error(banknote_train_examples, lambda sample: predict_standard(sample, weights_1))
    # 
    #     weights_2 = primal_svm(banknote_train_examples, epochs, learning_rate2(init2), C)
    #     test_err_2 = average_error(banknote_test_examples, lambda sample: predict_standard(sample, weights_2))
    #     train_err_2 = average_error(banknote_train_examples, lambda sample: predict_standard(sample, weights_2))
    # 
    #     weights_diff = weights_1-weights_2
    #     test_err_diff = test_err_1 - test_err_2
    #     train_err_diff = train_err_1 - train_err_2
    # 
    #     _1_string = ""
    #     _2_string = ""
    #     _diff_string = ""
    #     for i in range(len(weights_diff)):
    #         _1_string += "{:8.3f}\t&".format(weights_1[i])
    #         _2_string += "{:8.3f}\t&".format(weights_2[i])
    #         _diff_string += "{:8.3f}\t&".format(weights_diff[i])
    # 
    #     total_string = "{:8.3f}\t&".format(C)
    #     total_string += "LR1 &" + _1_string + "{:8.3f}\t&".format(train_err_1) + "{:8.3f}\t".format(test_err_1)
    #     total_string += "\n" + r"\\\cline{2-9}&"
    #     total_string += "LR2 &" +_2_string + "{:8.3f}\t&".format(train_err_2) + "{:8.3f}\t".format(test_err_2)
    #     total_string += "\n" + r"\\\cline{2-9}&"
    #     total_string += "DIFF &" +_diff_string + "{:8.3f}\t&".format(train_err_diff) + "{:8.3f}\t".format(test_err_diff)
    #     total_string += r"\\\hline\hline"
    # 
    # 
    #     print(total_string)
    # 
    # 

    train_labels = np.array(list(map(lambda sample: sample.label, banknote_train_examples)))
    train_vals = np.array(list(map(lambda sample: sample.attributes[:-1], banknote_train_examples)))

    # print(train_vals[:10])

    max_num_samples = 100
    print("TRAINING SUBSET SIZE: ", max_num_samples)

    for C in [100/873, 500/873, 700/873]:
        alphas = dual_svm(train_vals[:max_num_samples], train_labels[:max_num_samples], C)
        # print(alphas)
        incorporated_weights = recover_weights(alphas, train_vals[:max_num_samples], train_labels[:max_num_samples], 1e-11)

        train_err = average_error(banknote_train_examples[:max_num_samples],
                                  lambda sample: predict_standard(sample, incorporated_weights))
        test_err = average_error(banknote_train_examples[:max_num_samples],
                                 lambda sample: predict_standard(sample, incorporated_weights))
        other_train_err = average_error(banknote_train_examples[:max_num_samples],
                                  lambda sample: kernel_predict(alphas, train_vals[:max_num_samples], train_labels[:max_num_samples], np.dot, sample.attributes[:-1]))
        other_test_err = average_error(banknote_train_examples[:max_num_samples],
                             lambda sample: kernel_predict(alphas, train_vals[:max_num_samples], train_labels[:max_num_samples], np.dot, sample.attributes[:-1]))

        print(test_err, " ", other_test_err)
        print("DIFF: ", train_err - other_train_err, ", ", test_err - other_test_err)

        string = "{:8.3f}".format(C)
        for i in range(len(incorporated_weights)):
            string += "{:8.3f}\t&".format(incorporated_weights[i])

        print(string + r"\\\hline")
        # print(format_string.format(C, train_err, test_err))

    print()

    def kernel_classify(sample, a, x, y, gamma):
        def kerneling(w, x):
            return gaussian_kernel(w, x, gamma)
        return kernel_predict(alphas, x, y, kerneling, sample.attributes[:-1])



    # KERNELSTUFF

    format_string = "{:<7.3f}\t&{:<7.3f}\t&{:>10.3f}\t&{:>10.3f}\t\\\\\hline"

    for gamma in [10, 100]:
       for C in [100/873, 500/873, 700/873]:
            alphas = dual_svm(train_vals[:max_num_samples], train_labels[:max_num_samples], C, lambda w,x : gaussian_kernel(w,x, gamma))
            # print(alphas)
            # incorporated_weights = recover_weights(alphas, train_vals[:max_num_samples], train_labels[:max_num_samples], 1e-11)
            train_err = average_error(banknote_train_examples, lambda sample : kernel_classify(sample, alphas, train_vals[:max_num_samples], train_labels[:max_num_samples], gamma))
            test_err = average_error(banknote_test_examples, lambda sample : kernel_classify(sample, alphas, train_vals[:max_num_samples], train_labels[:max_num_samples], gamma))
            # train_err = average_error(banknote_train_examples[:max_num_samples], lambda sample : kernel_predict(alphas, train_vals[:max_num_samples], train_labels[:max_num_samples], lambda w,x:gaussian_kernel(w, x, gamma), sample.attributes[:-1]))
            # test_err = average_error(banknote_test_examples[:max_num_samples], lambda sample : kernel_predict(alphas, train_vals[:max_num_samples], train_labels[:max_num_samples], lambda w,x:gaussian_kernel(w, x, gamma), sample.attributes[:-1]))

            print(format_string.format(C, gamma, train_err, test_err))






    
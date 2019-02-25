
from DecisionTrees.TrainTree import *
from math import log, exp
import copy
from random import choices
from random import choice

WEIGHT_MIN = 1e-300


def compute_weighted_error(examples, classifier):
    incorrect = 0
    correct = 0
    total = 0
    for sample in examples:
        if classifier(sample) != sample.label:
            incorrect += sample.weight
        else:
            correct += sample.weight
        total += sample.weight
    return incorrect / total

def adaBoost_vote_weights_and_stumps(examples, num_stumps, categoricals):

    vote_weights = list()
    stumps = list()

    normalizer = 1/len(examples)
    for sample in examples:
        sample.weight *= normalizer

    for t in range(0, num_stumps):
        depth = 1
        stump = DecisionTree(depth, 0, examples, entropy_weighted, None, categoricals)
        error = compute_weighted_error(examples, stump.decide)

        depth = 2;
        while error > .5:
            stump = DecisionTree(depth, 0, examples, entropy_weighted, None, categoricals)
            error = compute_weighted_error(examples, stump.decide)
            depth +=1

        if error == 0:
            print("AH!")
        if error == 1:
            print("GAHHH")

        vote_weight = .5 * log((1-error)/error)
        total = 0
        for sample in examples:
            if stump.decide(sample) == sample.label:
                sample.weight *= exp(-vote_weight)
            else:
                sample.weight *= exp(vote_weight)
            total += sample.weight
        normalizer = 1/total
        for sample in examples:
            if sample.weight*normalizer != 0:
                sample.weight *= normalizer
            else:
                print(sample ," minned out!")
                sample.weight = WEIGHT_MIN
        vote_weights.append(vote_weight)
        stumps.append(stump)

    return vote_weights, stumps


def adaDecide(sample, vote_weights, stumps, num_to_consider):
    ballot = dict()
    leader = None
    leader_num = 0
    for t in range(0, num_to_consider):
        candidate = stumps[t].decide(sample)
        if candidate in ballot.keys():
            ballot[candidate] += sample.weight * vote_weights[t]
        else:
            ballot[candidate] = sample.weight * vote_weights[t]
        if ballot[candidate] > leader_num:
            leader_num = ballot[candidate]
            leader = candidate
    return leader

def baggy_trees(examples, subset_size, num_trees, categorics):
    trees = list()
    for tree_idx in range(0, num_trees):
        subset = pick_subset(examples, subset_size)
        trees.append(DecisionTree( len(examples[0].attributes), 0, subset, entropy_weighted, None, categorics) )
    return trees

def pick_subset(examples, subset_size):
    subset = [choice(examples) for _ in range(subset_size)]
    return subset

def bagging_decision(sample, trees, num_trees_to_use):
    ballot = dict()
    leader = None
    leader_num = 0
    for t in range(0, num_trees_to_use):
        candidate = trees[t].decide(sample)
        if candidate in ballot.keys():
            ballot[candidate] += sample.weight
        else:
            ballot[candidate] = sample.weight
        if ballot[candidate] > leader_num:
            leader_num = ballot[candidate]
            leader = candidate
    return leader

def baggy_data():
    print("Bags!")
    bank_train_file = "Data/bank/train.csv"
    bank_test_file = "Data/bank/test.csv"

    bank_train_examples_unknown_is_label, _, categoricals = examples_from_file(bank_train_file, True)
    bank_test_examples_unknown_is_label, _, categoricals = examples_from_file(bank_test_file, True)

    num_iters = 250
    trees = baggy_trees(bank_train_examples_unknown_is_label, 1000, num_iters, categoricals)

    header_string = "t,Bagging Training Error, Bagging Test Error"
    format_string = "{:<5},{:>10.7f},{:>10.7f}"
    print(header_string)
    for T in range(0,250):
        train_error = compute_weighted_error(bank_train_examples_unknown_is_label, lambda sample : bagging_decision(sample, trees, T*4))
        test_error = compute_weighted_error(bank_test_examples_unknown_is_label, lambda sample : bagging_decision(sample, trees, T*4))
        formatted = format_string.format(T*4, train_error, test_error)
        print(formatted)





def adaBoost_Data():
    print("AdaBoostin!")
    bank_train_file = "Data/bank/train.csv"
    bank_test_file = "Data/bank/test.csv"

    bank_train_examples_unknown_is_label, _, categoricals = examples_from_file(bank_train_file, True)
    bank_test_examples_unknown_is_label, _, categoricals = examples_from_file(bank_test_file, True)
    train_examples_copy = copy.deepcopy(bank_train_examples_unknown_is_label)

    num_iters = 1000

    votes, stumps = adaBoost_vote_weights_and_stumps(train_examples_copy, num_iters, categoricals)

    header_string = "t,AdaBoost Training Error, AdaBoost Test Error, Stump Training Error, Stump Test Error, Stump Depth"
    format_string = "{:<5},{:>10.7f},{:>10.7f},{:>10.7f},{:>10.7f},{:>5}"

    print(header_string)
    for T in range(1, num_iters):
        train_error = compute_weighted_error(bank_train_examples_unknown_is_label, lambda sample: adaDecide(sample, votes, stumps, T))
        test_error = compute_weighted_error(bank_test_examples_unknown_is_label, lambda sample: adaDecide(sample, votes, stumps, T))
        stump_train_error = compute_weighted_error(bank_train_examples_unknown_is_label, stumps[T].decide)
        stump_test_error = compute_weighted_error(bank_test_examples_unknown_is_label, stumps[T].decide)
        stump_depth = stumps[T].max_depth
        formatted = format_string.format(T,train_error,test_error,stump_train_error,stump_test_error,stump_depth)
        print(formatted)

if __name__ == '__main__':
    baggy_data()
    # adaBoost_Data()

from sys import path, argv, stdout
from os.path import dirname, realpath
MY_DIR = dirname(realpath(__file__))
path.append(MY_DIR)
PARENT_DIR = dirname(path[0])
path.append(PARENT_DIR)

import DecisionTrees.TrainTree
from math import log, exp, pow
import copy
from random import choice, sample as rnd_sample

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
        stump = DecisionTrees.TrainTree.DecisionTree(depth, 0, examples, DecisionTrees.TrainTree.entropy_weighted, None, categoricals)
        error = compute_weighted_error(examples, stump.decide)

        depth = 2;
        while error > .5:
            stump = DecisionTrees.TrainTree.DecisionTree(depth, 0, examples, DecisionTrees.TrainTree.entropy_weighted, None, categoricals)
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
        trees.append(DecisionTrees.TrainTree.DecisionTree(len(examples[0].attributes), 0, subset, DecisionTrees.TrainTree.entropy_weighted, None, categorics))
    return trees

def baggy_forest(examples, attribute_subset_size, example_subset_size, num_trees, categorics):
    trees = list()
    for tree_idx in range(0, num_trees):
        subset = pick_subset(examples, example_subset_size)
        trees.append(DecisionTrees.TrainTree.DecisionTree(len(examples[0].attributes), 0, subset, DecisionTrees.TrainTree.entropy_weighted, None, categorics, True, attribute_subset_size))
    return trees

def baggy_forests(examples, example_subset_size, num_trees, categorics):
    trees_2 = list()
    trees_4 = list()
    trees_6 = list()
    num_attributes = len(examples[0].attributes)
    marksize = (int) (num_trees/100)
    for tree_idx in range(0, num_trees):
        if tree_idx % marksize == 0:
            stdout.write(".")
            stdout.flush()
        subset = pick_subset(examples, example_subset_size)
        trees_2.append(DecisionTrees.TrainTree.DecisionTree(num_attributes, 0, subset, DecisionTrees.TrainTree.entropy_weighted, None, categorics, True, 2))
        trees_4.append(DecisionTrees.TrainTree.DecisionTree(num_attributes, 0, subset, DecisionTrees.TrainTree.entropy_weighted, None, categorics, True, 4))
        trees_6.append(DecisionTrees.TrainTree.DecisionTree(num_attributes, 0, subset, DecisionTrees.TrainTree.entropy_weighted, None, categorics, True, 6))
    return trees_2, trees_4, trees_6

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

    bank_train_examples, _, categoricals = DecisionTrees.TrainTree.examples_from_file(bank_train_file, True)
    bank_test_examples, _, categoricals = DecisionTrees.TrainTree.examples_from_file(bank_test_file, True)

    trees = baggy_trees(bank_train_examples, 500, 1000, categoricals)

    print("0 to 1000 in increments of 20")
    header_string = "t,Bagging Training Error, Bagging Test Error"
    format_string = "{:<5},{:>10.7f},{:>10.7f}"
    print(header_string)
    for T in range(0,1000, 10):
        train_error = compute_weighted_error(bank_train_examples, lambda sample : bagging_decision(sample, trees, T))
        test_error = compute_weighted_error(bank_test_examples, lambda sample : bagging_decision(sample, trees, T))
        formatted = format_string.format(T, train_error, test_error)
        print(formatted)

def forest_data():
    print("Forests!")
    bank_train_file = MY_DIR + "/Data/bank/train.csv"
    bank_test_file = MY_DIR + "/Data/bank/test.csv"

    bank_train_examples, _, categoricals = DecisionTrees.TrainTree.examples_from_file(bank_train_file, True)
    bank_test_examples, _, categoricals = DecisionTrees.TrainTree.examples_from_file(bank_test_file, True)

    # trees_2 = baggy_forest(bank_train_examples, 2, 500, 1000, categoricals)
    # trees_4 = baggy_forest(bank_train_examples, 4, 500, 1000, categoricals)
    # trees_6 = baggy_forest(bank_train_examples, 6, 500, 1000, categoricals)
    print("|         |         |         |         |         |         |         |         |         |        |")
    trees_2, trees_4, trees_6 = baggy_forests(bank_test_examples, 500, 1000, categoricals)
    print("Forests loaded!")

    compute_forest_errors(bank_train_examples, bank_test_examples, trees_2, trees_4, trees_6)
    # print("0 to 1000 in increments of 20")
    # header_string = "t,Random Forest 2 Training Error, Random Forest 2 Test Error,Random Forest 4 Training Error, Random Forest 4 Test Error,Random Forest 6 Training Error, Random Forest 6 Test Error"
    # format_string = "{:<5},{:>10.7f},{:>10.7f},{:>10.7f},{:>10.7f},{:>10.7f},{:>10.7f}"
    # print(header_string)
    # for T in range(0, 1000, 20):
    #     train_error_2 = compute_weighted_error(bank_train_examples, lambda sample : bagging_decision(sample, trees_2, T))
    #     test_error_2 = compute_weighted_error(bank_test_examples, lambda sample : bagging_decision(sample, trees_2, T))
    #     train_error_4 = compute_weighted_error(bank_train_examples, lambda sample : bagging_decision(sample, trees_4, T))
    #     test_error_4 = compute_weighted_error(bank_test_examples, lambda sample : bagging_decision(sample, trees_4, T))
    #     train_error_6 = compute_weighted_error(bank_train_examples, lambda sample : bagging_decision(sample, trees_6, T))
    #     test_error_6 = compute_weighted_error(bank_test_examples, lambda sample : bagging_decision(sample, trees_6, T))
    #     formatted = format_string.format(T, train_error_2, test_error_2, train_error_4, test_error_4, train_error_6, test_error_6)
    #     print(formatted)


def compute_forest_errors(train_examples, test_examples, trees_2, trees_4, trees_6):
    header_string = "t,Random Forest 2 Training Error, Random Forest 2 Test Error,Random Forest 4 Training Error, Random Forest 4 Test Error,Random Forest 6 Training Error, Random Forest 6 Test Error"
    format_string = "{:<5},{:>10.7f},{:>10.7f},{:>10.7f},{:>10.7f},{:>10.7f},{:>10.7f}"
    print(header_string)
    for T in range(0, 1000,10):
        trees_2_incorrect = trees_4_incorrect = trees_6_incorrect = total = 0
        for sample in train_examples:
            if bagging_decision(sample, trees_2, T) != sample.label:
                trees_2_incorrect += sample.weight
            if bagging_decision(sample, trees_4, T) != sample.label:
                trees_4_incorrect += sample.weight
            if bagging_decision(sample, trees_6, T) != sample.label:
                trees_6_incorrect += sample.weight
            total += sample.weight
        train_error_2 = trees_2_incorrect / total
        test_error_2 = trees_4_incorrect / total
        train_error_4 = trees_6_incorrect / total

        for sample in test_examples:
            if bagging_decision(sample, trees_2, T) != sample.label:
                trees_2_incorrect += sample.weight
            if bagging_decision(sample, trees_4, T) != sample.label:
                trees_4_incorrect += sample.weight
            if bagging_decision(sample, trees_6, T) != sample.label:
                trees_6_incorrect += sample.weight
            total += sample.weight

        test_error_4 = trees_2_incorrect / total
        train_error_6 = trees_4_incorrect / total
        test_error_6 = trees_6_incorrect / total

        formatted = format_string.format(T, train_error_2, test_error_2, train_error_4, test_error_4, train_error_6, test_error_6)
        print(formatted)



def adaBoost_Data():
    print("AdaBoostin!")
    bank_train_file = MY_DIR + "/Data/bank/train.csv"
    bank_test_file = MY_DIR + "/Data/bank/test.csv"

    bank_train_examples, _, categoricals = DecisionTrees.TrainTree.examples_from_file(bank_train_file, True)
    bank_test_examples, _, categoricals = DecisionTrees.TrainTree.examples_from_file(bank_test_file, True)
    train_examples_copy = copy.deepcopy(bank_train_examples)

    num_iters = 1000

    votes, stumps = adaBoost_vote_weights_and_stumps(train_examples_copy, num_iters, categoricals)

    header_string = "t,AdaBoost Training Error, AdaBoost Test Error, Stump Training Error, Stump Test Error, Stump Depth"
    format_string = "{:<5},{:>10.7f},{:>10.7f},{:>10.7f},{:>10.7f},{:>5}"
    print(header_string)
    for T in range(1, num_iters, 10):
        train_error = compute_weighted_error(bank_train_examples, lambda sample: adaDecide(sample, votes, stumps, T))
        test_error = compute_weighted_error(bank_test_examples, lambda sample: adaDecide(sample, votes, stumps, T))
        stump_train_error = compute_weighted_error(bank_train_examples, stumps[T].decide)
        stump_test_error = compute_weighted_error(bank_test_examples, stumps[T].decide)
        stump_depth = stumps[T].max_depth
        formatted = format_string.format(T,train_error,test_error,stump_train_error,stump_test_error,stump_depth)
        print(formatted)

def ALL_THE_BAGS():
    print("Prepare to wait...")
    bank_train_file = MY_DIR + "/Data/bank/train.csv"
    bank_test_file = MY_DIR + "/Data/bank/test.csv"

    bank_train_examples, _, categoricals = DecisionTrees.TrainTree.examples_from_file(bank_train_file,True)
    bank_test_examples, _, categoricals = DecisionTrees.TrainTree.examples_from_file(bank_test_file,True)

    bags_o_trees = list()
    bags_o_forests = list()
    NUM_TREES_IN_BAG = 1000
    print("Bag size: ", NUM_TREES_IN_BAG, "Progress bar: ")
    print("|         |         |         |         |         |         |         |         |         |        |")
    for run in range(0, 100):
        stdout.write(".")
        stdout.flush()
        subset = rnd_sample(bank_train_examples, 1000)
        bags_o_trees.append(baggy_trees(subset, 100, NUM_TREES_IN_BAG, categoricals))
        bags_o_forests.append(baggy_forest(subset, 100, 4, NUM_TREES_IN_BAG, categoricals))
    total_bag_variance = 0
    total_bag_bias = 0
    total_tree_variance = 0
    total_tree_bias = 0
    total_forest_variance = 0
    total_forest_bias = 0
    total_rnd_tree_variance = 0
    total_rnd_tree_bias = 0

    print("\nTrained the trees, finally! On to computing ALL THE BIAS STUFF")
    print("Number of samples: ", len(bank_test_examples), "Progress bar: ")
    print("|         |         |         |         |         |         |         |         |         |        |")
    done = 0
    marksize = (int)(len(bank_test_examples) / 100)
    for sample in bank_test_examples:
        if done % marksize == 0:
            stdout.write(".")
            stdout.flush()

        bag_average = 0
        bag_variance = 0
        tree_average = 0
        tree_variance = 0
        forest_average = 0
        forest_variance = 0
        rnd_tree_average = 0
        rnd_tree_variance = 0


        for i in range(0, 100):
            if bagging_decision(sample, bags_o_trees[i], NUM_TREES_IN_BAG) != "no":
                bag_average += 1
            if bagging_decision(sample, bags_o_forests[i], NUM_TREES_IN_BAG) != "no":
                forest_average += 1
            if bags_o_trees[i][0].decide(sample) != "no":
                tree_average += 1
            if bags_o_forests[i][0].decide(sample) != "no":
                rnd_tree_average += 1

        bag_average /= 100
        tree_average /= 100
        forest_average /= 100
        rnd_tree_average /= 100
        for i in range(0, 100):
            if bagging_decision(sample, bags_o_trees[i], NUM_TREES_IN_BAG) == "no":
                bag_variance += bag_average * bag_average
            else:
                bag_variance += (1 - bag_average) * (1 - bag_average)
            if bagging_decision(sample, bags_o_forests[i], NUM_TREES_IN_BAG) == "no":
                forest_variance += forest_average * forest_average
            else:
                forest_variance += (1-forest_average) * (1-forest_average)
            if bags_o_trees[i][0].decide(sample) == "no":
                tree_variance += tree_average * tree_average
            else:
                tree_variance += (1-tree_average)*(1-tree_average)
            if bags_o_forests[i][0].decide(sample) == "no":
                rnd_tree_variance += rnd_tree_average * rnd_tree_average
            else:
                rnd_tree_variance += (1 - rnd_tree_average) * (1 - rnd_tree_average)

        bag_variance /= 99
        tree_variance /= 99
        forest_variance /= 99
        rnd_tree_variance /= 99

        # bag_average = sum(0 if bagging_decision(sample, bag,NUM_TREES_IN_BAG)=="no" else 1 for bag in bags_o_trees) / 100
        # bag_variance = sum( pow( ((0 if bagging_decision(sample, bag,NUM_TREES_IN_BAG)=="no" else 1) - bag_average), 2) for bag in bags_o_trees) / 99
        # tree_average = sum(0 if bag[0].decide(sample) == "no" else 1 for bag in bags_o_trees) / 100
        # tree_variance = sum(pow(((0 if bag[0].decide(sample) == "no" else 1) - tree_average), 2) for bag in bags_o_trees) / 99
        # forest_average = sum(0 if bagging_decision(sample, bag,NUM_TREES_IN_BAG)=="no" else 1 for bag in bags_o_forests) / 100
        # forest_variance = sum( pow( ((0 if bagging_decision(sample, bag,NUM_TREES_IN_BAG)=="no" else 1) - forest_average), 2) for bag in bags_o_forests) / 99
        # rnd_tree_average = sum(0 if bag[0].decide(sample) == "no" else 1 for bag in bags_o_forests) / 100
        # rnd_tree_variance = sum(pow(((0 if bag[0].decide(sample) == "no" else 1) - rnd_tree_average), 2) for bag in bags_o_forests) / 99


        bag_bias = pow( bag_average - (0 if sample.label=="no" else 1) , 2)

        total_bag_variance += bag_variance
        total_bag_bias += bag_bias

        tree_bias = pow(tree_average - (0 if sample.label == "no" else 1), 2)

        total_tree_bias += tree_bias
        total_tree_variance += tree_variance


        forest_bias = pow( forest_average - (0 if sample.label=="no" else 1) , 2)

        total_forest_variance += forest_variance
        total_forest_bias += forest_bias


        rnd_tree_bias = pow(rnd_tree_average - (0 if sample.label == "no" else 1), 2)

        total_rnd_tree_variance += rnd_tree_variance
        total_rnd_tree_bias += rnd_tree_bias

        done += 1

    num_examples = len(bank_train_examples)
    total_bag_variance /= num_examples
    total_bag_bias /= num_examples
    total_tree_variance /= num_examples
    total_tree_bias /= num_examples
    total_forest_variance /= num_examples
    total_forest_bias /= num_examples
    total_rnd_tree_variance /= num_examples
    total_rnd_tree_bias /= num_examples

    format_string = "{:20}\t&{:>10.4f}\t&{:>10.4f}\t&{:>10.4f}\t\\\\\hline"
    print("\n\nMethod \t& Bias \t& Variance \t& General Error\t \\\\\hline")
    print(format_string.format("Bagging", total_bag_bias, total_bag_variance, total_bag_bias+total_bag_variance))
    print(format_string.format("Random Forest", total_forest_bias, total_forest_variance, total_forest_bias+total_bag_variance))
    print(format_string.format("Single Tree", total_tree_bias, total_tree_variance, total_tree_bias+total_tree_variance))
    print(format_string.format("Single Randomized Tree", total_rnd_tree_bias, total_rnd_tree_variance, total_rnd_tree_bias+total_rnd_tree_variance))


def ALL_THE_BAGS2():
    print("Prepare to wait for slightly less time!")
    bank_train_file = MY_DIR + "/Data/bank/train.csv"
    bank_test_file = MY_DIR + "/Data/bank/test.csv"

    bank_train_examples, _, categoricals = DecisionTrees.TrainTree.examples_from_file(bank_train_file,True)
    bank_test_examples, _, categoricals = DecisionTrees.TrainTree.examples_from_file(bank_test_file,True)

    bags_o_trees = list()
    bags_o_forests = list()
    bag_predictions = list()
    forest_predictions = list()
    tree_predictions = list()
    rnd_tree_predictions = list()
    NUM_TREES_IN_BAG = 10
    print("Bag size: ", NUM_TREES_IN_BAG, "Progress bar: ")
    print("|         |         |         |         |         |         |         |         |         |        |")
    for run in range(0, 100):
        stdout.write(".")
        stdout.flush()
        subset = rnd_sample(bank_train_examples, 1000)
        bag = baggy_trees(subset, 100, NUM_TREES_IN_BAG, categoricals)
        forest = baggy_forest(subset, 100, 4, NUM_TREES_IN_BAG, categoricals)
        bag_predictions.append(list())
        forest_predictions.append(list())
        tree_predictions.append(list())
        rnd_tree_predictions.append(list())
        for sample in bank_test_examples:
            bag_predictions[-1].append(0 if bagging_decision(sample,bag,NUM_TREES_IN_BAG)=="no" else 1)
            forest_predictions[-1].append(0 if bagging_decision(sample, forest, NUM_TREES_IN_BAG)=="no" else 1)
            tree_predictions[-1].append(0 if bag[0].decide(sample)=="no" else 1)
            rnd_tree_predictions[-1].append(0 if forest[0].decide(sample)=="no" else 1)

    total_bag_variance = 0
    total_bag_bias = 0
    total_tree_variance = 0
    total_tree_bias = 0
    total_forest_variance = 0
    total_forest_bias = 0
    total_rnd_tree_variance = 0
    total_rnd_tree_bias = 0

    print("\nTrained the trees, finally! On to computing ALL THE BIAS STUFF")
    print("Number of samples: ", len(bank_test_examples), "Progress bar: ")
    print("|         |         |         |         |         |         |         |         |         |        |")
    done = 0
    marksize = (int)(len(bank_test_examples) / 100)
    for sample in bank_test_examples:
        if done % marksize == 0:
            stdout.write(".")
            stdout.flush()

        bag_average = 0
        bag_variance = 0
        tree_average = 0
        tree_variance = 0
        forest_average = 0
        forest_variance = 0
        rnd_tree_average = 0
        rnd_tree_variance = 0


        # for i in range(0, 100):
        #     if bagging_decision(sample, bags_o_trees[i], NUM_TREES_IN_BAG) != "no":
        #         bag_average += 1
        #     if bagging_decision(sample, bags_o_forests[i], NUM_TREES_IN_BAG) != "no":
        #         forest_average += 1
        #     if bags_o_trees[i][0].decide(sample) != "no":
        #         tree_average += 1
        #     if bags_o_forests[i][0].decide(sample) != "no":
        #         rnd_tree_average += 1
        #
        # bag_average /= 100
        # tree_average /= 100
        # forest_average /= 100
        # rnd_tree_average /= 100
        # for i in range(0, 100):
        #     if bagging_decision(sample, bags_o_trees[i], NUM_TREES_IN_BAG) == "no":
        #         bag_variance += bag_average * bag_average
        #     else:
        #         bag_variance += (1 - bag_average) * (1 - bag_average)
        #     if bagging_decision(sample, bags_o_forests[i], NUM_TREES_IN_BAG) == "no":
        #         forest_variance += forest_average * forest_average
        #     else:
        #         forest_variance += (1-forest_average) * (1-forest_average)
        #     if bags_o_trees[i][0].decide(sample) == "no":
        #         tree_variance += tree_average * tree_average
        #     else:
        #         tree_variance += (1-tree_average)*(1-tree_average)
        #     if bags_o_forests[i][0].decide(sample) == "no":
        #         rnd_tree_variance += rnd_tree_average * rnd_tree_average
        #     else:
        #         rnd_tree_variance += (1 - rnd_tree_average) * (1 - rnd_tree_average)
        #
        # bag_variance /= 99
        # tree_variance /= 99
        # forest_variance /= 99
        # rnd_tree_variance /= 99

        bag_average = sum(bag_pred[done] for bag_pred in bag_predictions) / 100
        bag_variance = sum( pow( (bag_pred[done] - bag_average), 2) for bag_pred in bag_predictions) / 99
        tree_average = sum(tree_pred[done] for tree_pred in tree_predictions) / 100
        tree_variance = sum( pow(tree_pred[done] - tree_average, 2) for tree_pred in tree_predictions) / 99
        forest_average = sum(for_pred[done] for for_pred in forest_predictions) / 100
        forest_variance = sum( pow( (for_pred[done] - forest_average), 2) for for_pred in forest_predictions) / 99
        rnd_tree_average = sum(rnd_tree_pred[done] for rnd_tree_pred in rnd_tree_predictions) / 100
        rnd_tree_variance = sum(pow((rnd_tree_pred[done] - rnd_tree_average), 2) for rnd_tree_pred in rnd_tree_predictions) / 99


        bag_bias = pow( bag_average - (0 if sample.label=="no" else 1) , 2)

        total_bag_variance += bag_variance
        total_bag_bias += bag_bias

        tree_bias = pow(tree_average - (0 if sample.label == "no" else 1), 2)

        total_tree_bias += tree_bias
        total_tree_variance += tree_variance


        forest_bias = pow( forest_average - (0 if sample.label=="no" else 1) , 2)

        total_forest_variance += forest_variance
        total_forest_bias += forest_bias


        rnd_tree_bias = pow(rnd_tree_average - (0 if sample.label == "no" else 1), 2)

        total_rnd_tree_variance += rnd_tree_variance
        total_rnd_tree_bias += rnd_tree_bias

        done += 1

    num_examples = len(bank_train_examples)
    total_bag_variance /= num_examples
    total_bag_bias /= num_examples
    total_tree_variance /= num_examples
    total_tree_bias /= num_examples
    total_forest_variance /= num_examples
    total_forest_bias /= num_examples
    total_rnd_tree_variance /= num_examples
    total_rnd_tree_bias /= num_examples

    format_string = "{:20}\t&{:>10.4f}\t&{:>10.4f}\t&{:>10.4f}\t\\\\\hline"
    print("\n\nMethod \t& Bias \t& Variance \t& General Error\t \\\\\hline")
    print(format_string.format("Bagging", total_bag_bias, total_bag_variance, total_bag_bias+total_bag_variance))
    print(format_string.format("Random Forest", total_forest_bias, total_forest_variance, total_forest_bias+total_bag_variance))
    print(format_string.format("Single Tree", total_tree_bias, total_tree_variance, total_tree_bias+total_tree_variance))
    print(format_string.format("Single Randomized Tree", total_rnd_tree_bias, total_rnd_tree_variance, total_rnd_tree_bias+total_rnd_tree_variance))


if __name__ == '__main__':

    if len(argv) < 2:
        # baggy_data()
        ALL_THE_BAGS2()
        # forest_data()
        # adaBoost_Data()
    else:
        which = argv[1]
        if which == "bag":
            print("Lists of bags")
            baggy_data()
        elif which == "somuchstuff":
            print("All dem bags")
            ALL_THE_BAGS2()
        elif which == "forest":
            print("Forests for dayyyyz")
            forest_data()
        elif which == "adaBoost":
            print("I'm into that Ada's Boost")
            adaBoost_Data()
        else:
            # baggy_data()
            print("All dem bags")
            ALL_THE_BAGS2()
            # forest_data()
            # adaBoost_Data()


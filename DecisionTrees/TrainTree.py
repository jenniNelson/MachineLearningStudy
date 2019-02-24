

from collections import Counter
from statistics import median
from math import log

class Example:
    def __init__(self, values, weight=1):
        self.attributes = values[0: len(values)-1]
        self.label = values[len(values)-1]
        self.weight = weight


class DecisionTree:

    def __init__(self, max_depth, depth, examples, gain_metric, most_common_label, is_categoric_attribute):
        self.depth = depth
        self.max_depth = max_depth

        if len(examples) == 0:
            self.decide = lambda sample : most_common_label
            return

        most_common_label = DecisionTree.most_common_label(examples)
        if depth == max_depth:
            self.decide = lambda sample : most_common_label
            return


        best_attribute = DecisionTree.find_best_attribute(gain_metric, examples, is_categoric_attribute)

        if is_categoric_attribute[best_attribute]:
            # partition examples
            categorical_partitions = partition_examples_categorically(examples, best_attribute)

            self.categorical_children = dict()
            for value in categorical_partitions.keys():
                self.categorical_children[value] = DecisionTree(max_depth, depth + 1, categorical_partitions[value], gain_metric, most_common_label, is_categoric_attribute)

            def categorical_decision(sample):
                if sample.attributes[best_attribute] in self.categorical_children.keys():
                    return self.categorical_children[sample.attributes[best_attribute]].decide(sample)
                else:
                    return most_common_label

            self.decide = categorical_decision
        else:

            # partition examples numerically:
            less_than_examples, greater_than_examples, threshold = partition_examples_numerically(examples, best_attribute)

            self.less_child = DecisionTree(max_depth, depth + 1, less_than_examples, gain_metric, most_common_label, is_categoric_attribute)
            self.greater_child = DecisionTree(max_depth, depth + 1, greater_than_examples, gain_metric,
                                              most_common_label, is_categoric_attribute)

            def numeric_decision(sample):
                if isinstance(sample.attributes[best_attribute], float):
                    if threshold < sample.attributes[best_attribute]:
                        return self.greater_child.decide(sample)
                    else:
                        return self.less_child.decide(sample)
                else:
                    return most_common_label

            self.decide = numeric_decision





    @classmethod
    def find_best_attribute(cls, gain_metric, examples, is_numeric_attributes):
        best_so_far = -1.0
        best_gain_val_so_far = float("-inf")
        for attribute_idx in range(len(examples[0].attributes)):
            gain = info_gain(examples, attribute_idx, is_numeric_attributes, gain_metric)
            if gain > best_gain_val_so_far:
                best_so_far = attribute_idx
                best_gain_val_so_far = gain
        return best_so_far


    @classmethod
    def most_common_label(cls, examples):
        labels = list()
        for sample in examples:
            labels.append(sample.label)
        return Counter(labels).most_common(1)[0][0]



def info_gain(examples, attribute, is_categoric_attributes, gain_metric):

    num_examples = len(examples)
    if num_examples == 0:
        return 0

    gain = gain_metric(examples)

    if is_categoric_attributes[attribute]:
        categoric_partitions = partition_examples_categorically(examples, attribute)

        for partition in categoric_partitions.values():
            gain -= len(partition)/num_examples * gain_metric(partition)

        return gain
    else:
        less_subset, greater_subset, tmp = partition_examples_numerically(examples, attribute)
        return gain - len(less_subset)/num_examples * gain_metric(less_subset) \
                    - len(greater_subset)/num_examples * gain_metric(greater_subset)




def entropy(examples):
    if len(examples) == 0:
        return 0
    label_counts = dict()
    total_count = 0
    for sample in examples:
        total_count += 1
        if sample.label in label_counts:
            label_counts[sample.label] += 1
        else:
            label_counts[sample.label] = 1

    entropy_val = 0
    for count in label_counts.values():
        if count == 0:
            continue
        proportion = count / total_count
        entropy_val += -proportion * log(proportion)

    return entropy_val

def entropy_weighted(examples):
    if len(examples) == 0:
        return 0
    counts, most_common, total_count = weighted_counts_and_most_common(examples)

    entropy_val = 0
    for label, count in counts.items():
        if count == 0:
            continue
        proportion = count / total_count
        entropy_val += -proportion * log(proportion)

    return entropy_val

def majority_error(examples):
    if len(examples) == 0:
        return 0
    labels = (map(lambda samp: samp.label, examples))

    num_most_common = Counter(labels).most_common(1)[0][1]

    return 1 - num_most_common/len(examples)

def majority_weighted(examples):
    if len(examples) == 0:
        return 0
    counts, most_common, total_count = weighted_counts_and_most_common(examples)

    return 1 - counts[most_common]/total_count

def gini_index(examples):
    if len(examples) == 0:
        return 0
    labels = (map(lambda samp: samp.label, examples))

    gini = 1
    num_examples = len(examples)
    for label, count in Counter(labels).most_common():
        gini -= (count/num_examples)*(count/num_examples)

    return gini

def gini_weighted(examples):
    if len(examples) == 0:
        return 0
    counts, most_common, total_count = weighted_counts_and_most_common(examples)

    gini = 1
    for label, count in counts.items():
        portion = count / total_count
        gini -= portion*portion
    return gini


def weighted_counts_and_most_common(examples):
    if len(examples) == 0:
        return 0
    counts = dict()
    total_count = 0
    most_common_number = 0
    most_common_label = None
    for sample in examples:
        total_count += sample.weight
        if sample.label in counts:
            counts[sample.label] += sample.weight
        else:
            counts[sample.label] = sample.weight
        if counts[sample.label] > most_common_number:
            most_common_label = sample.label
            most_common_number = counts[sample.label]
    return counts, most_common_label, total_count



def partition_examples_categorically(examples, partitioning_attribute):
    categorical_partitions = dict()
    for sample in examples:
        if sample.attributes[partitioning_attribute] not in categorical_partitions.keys():
            categorical_partitions[sample.attributes[partitioning_attribute]] = [sample]
        else:
            categorical_partitions[sample.attributes[partitioning_attribute]].append(sample)
    return categorical_partitions

def partition_examples_numerically(examples, partitioning_attribute):
    less_than_examples, greater_than_examples = list(), list()
    split = median(list(map(lambda x : x.attributes[partitioning_attribute], examples)))
    for sample in examples:
        if sample.attributes[partitioning_attribute] < split:
            less_than_examples.append(sample)
        else:
            greater_than_examples.append(sample)
    return less_than_examples, greater_than_examples, split


def examples_from_attributes(file, unknown_is_label):
    examples = list()
    num_attributes = None
    attribute_values = list()
    is_categoric_attribute = list()
    most_common_values = dict()

    with open(file, 'r') as train_data:
        for line in train_data:
            terms = line.strip().split(',')
            if num_attributes is None:
                num_attributes = len(terms) - 1
                for idx in range(num_attributes):
                    attribute_values.append(set())
                    is_categoric_attribute.append(False)

            elif num_attributes != len(terms)-1 :
                raise ValueError("Example has an unexpected number of attributes.")

            for idx in range(num_attributes):
                try:
                    terms[idx] = float(terms[idx])
                except ValueError:
                    is_categoric_attribute[idx] = True
                attribute_values[idx].add(terms[idx])
            else:
                sample = Example(terms)
                examples.append(sample)

    for attribute in range(num_attributes):
        if is_categoric_attribute[attribute]:
            most_common = str
            count = Counter(attribute_values[attribute])
            most_common = count.most_common(1)[0][0]
            if not unknown_is_label and most_common == "unknown":  # If the most common was unknown, use second-most common
                most_common = count.most_common(2)[0][0]
            most_common_values[attribute] = most_common
            attribute_values[attribute] = set(attribute_values[attribute])
        else:
            attribute_values[attribute] = median(attribute_values[attribute])

    if not unknown_is_label:
        for sample in examples:
            for attribute in range(num_attributes):
                if sample.attributes[attribute] == "unknown":
                    sample.attributes[attribute] = most_common_values[attribute]
    return examples, attribute_values, is_categoric_attribute


def average_error(tree, test_examples):

    incorrect = 0
    correct = 0
    for test in test_examples:
        label = tree.decide(test)
        if test.label != label:
            incorrect += 1
        else:
            correct += 1
    return incorrect / (incorrect + correct)


if __name__ == '__main__':
    car_train_file = "Data/car/train.csv"
    car_test_file = "Data/car/test.csv"
    bank_train_file = "Data/bank/train.csv"
    bank_test_file = "Data/bank/test.csv"



    car_train_examples, foo, categoricals = examples_from_attributes(car_train_file, True)
    car_train_examples_unknown_not_label, foo, categoricals = examples_from_attributes(car_train_file, False)

    car_test_examples, foo, categoricals = examples_from_attributes(car_test_file, True)



    format_string = "{:<5} & {:>8.3f} & {:>8.3f} & {:>8.3f} & {:>8.3f} & {:>8.3f} & {:>8.3f}\t\\\\\t\\hline"
    format_string2 = "{:<5} & {:>8} & {:>8} & {:>8} & {:>8} & {:>8} & {:>8}\t\\\\\t\\hline\hline"

    print(r"\newcolumntype{R}{>{\centering\arraybackslash}X}")
    print(r"\begin{tabularx}{0.75\textwidth}{c||cc|cc|cc}")
    print(r"\multicolumn{7}{c}{\bf{Car Dataset Prediction Errors}} \\ \hline")
    print( r"& \multicolumn{2}{c||}{\bf{Entropy}} &\multicolumn{2}{c||}{\bf{Majority Error}} &\multicolumn{2}{c}{\bf{Gini Index}} \\ \hline")
    print(format_string2.format("Depth","Train", "Test", "Train", "Test", "Train", "Test"))
    for depth in range(1,7):

        EN_tree = DecisionTree(depth, 0, car_train_examples, entropy_weighted, None, categoricals)
        ME_tree = DecisionTree(depth, 0, car_train_examples, majority_weighted, None, categoricals)
        GI_tree = DecisionTree(depth, 0, car_train_examples, gini_weighted, None, categoricals)

        EN_train_err = average_error(EN_tree, car_train_examples)
        EN_test_err = average_error(EN_tree, car_test_examples)
        ME_train_err = average_error(ME_tree, car_train_examples)
        ME_test_err = average_error(ME_tree, car_test_examples)
        GI_train_err = average_error(GI_tree, car_train_examples)
        GI_test_err = average_error(GI_tree, car_test_examples)

        formatted = format_string.format(depth, EN_train_err, EN_test_err, ME_train_err, ME_test_err, GI_train_err, GI_test_err)
        print(formatted)
    print(r"\end{tabularx}")

    print()
    print()

    bank_test_examples_unknown_is_label, foo, categoricals = examples_from_attributes(bank_test_file, True)
    bank_train_examples_unknown_is_label, foo, categoricals = examples_from_attributes(bank_train_file, True)

    print(r"\newcolumntype{R}{>{\centering\arraybackslash}X}")
    print(r"\begin{tabularx}{0.75\textwidth}{c||cc|cc|cc}")
    print(r"\multicolumn{7}{c}{\bf{Bank Dataset Prediction Errors, Not Replacing Unknowns}} \\ \hline")
    print(r"& \multicolumn{2}{R||}{\bf{Entropy}} &\multicolumn{2}{R||}{\bf{Majority Error}} &\multicolumn{2}{R}{\bf{Gini Index}} \\ \hline")
    print(format_string2.format("Depth", "Train", "Test", "Train", "Test", "Train", "Test"))
    for depth in range(1, 17):
        EN_tree = DecisionTree(depth, 0, bank_train_examples_unknown_is_label, entropy_weighted, None, categoricals)
        ME_tree = DecisionTree(depth, 0, bank_train_examples_unknown_is_label, majority_weighted, None, categoricals)
        GI_tree = DecisionTree(depth, 0, bank_train_examples_unknown_is_label, gini_weighted, None, categoricals)

        EN_train_err = average_error(EN_tree, bank_train_examples_unknown_is_label)
        EN_test_err = average_error(EN_tree, bank_test_examples_unknown_is_label)
        ME_train_err = average_error(ME_tree, bank_train_examples_unknown_is_label)
        ME_test_err = average_error(ME_tree, bank_test_examples_unknown_is_label)
        GI_train_err = average_error(GI_tree, bank_train_examples_unknown_is_label)
        GI_test_err = average_error(GI_tree, bank_test_examples_unknown_is_label)

        formatted = format_string.format(depth, EN_train_err, EN_test_err, ME_train_err, ME_test_err,
                                         GI_train_err, GI_test_err)
        print(formatted)
    print(r"\end{tabularx}")





    bank_train_examples_unknown_not_label, foo, categoricals = examples_from_attributes(bank_train_file, False)

    print()
    print()

    print(r"\newcolumntype{R}{>{\centering\arraybackslash}X}")
    print(r"\begin{tabularx}{0.75\textwidth}{c||cc|cc|cc}")
    print(r"\multicolumn{7}{c}{\bf{Bank Dataset Prediction Errors, Replacing Unknowns}} \\ \hline")
    print(r"& \multicolumn{2}{R||}{\bf{Entropy}} &\multicolumn{2}{R||}{\bf{Majority Error}} &\multicolumn{2}{R}{\bf{Gini Index}} \\ \hline")
    print(format_string2.format("Depth", "Train", "Test", "Train", "Test", "Train", "Test"))
    for depth in range(1, 17):
        ME_tree = DecisionTree(depth, 0, bank_train_examples_unknown_not_label, entropy_weighted, None, categoricals)
        GI_tree = DecisionTree(depth, 0, bank_train_examples_unknown_not_label, majority_weighted, None, categoricals)
        EN_tree = DecisionTree(depth, 0, bank_train_examples_unknown_not_label, gini_weighted, None, categoricals)

        EN_train_err = average_error(EN_tree, bank_train_examples_unknown_not_label)
        EN_test_err = average_error(EN_tree, bank_test_examples_unknown_is_label)
        ME_train_err = average_error(ME_tree, bank_train_examples_unknown_not_label)
        ME_test_err = average_error(ME_tree, bank_test_examples_unknown_is_label)
        GI_train_err = average_error(GI_tree, bank_train_examples_unknown_not_label)
        GI_test_err = average_error(GI_tree, bank_test_examples_unknown_is_label)

        formatted = format_string.format(depth, EN_train_err, EN_test_err, ME_train_err, ME_test_err,
                                         GI_train_err, GI_test_err)
        print(formatted)
    print(r"\end{tabularx}")


# main()
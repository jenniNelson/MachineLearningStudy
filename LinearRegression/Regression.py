

class Example:
    def __init__(self, values, weight=1):
        self.attributes = values[0: len(values)-1]
        self.label = values[len(values)-1]
        self.weight = weight


def examples_from_file(filename):
    examples = list()
    num_attributes = None
    is_categoric_attribute = list()
    most_common_values = dict()

    with open(file, 'r') as train_data:
        for line in train_data:
            terms = line.strip().split(',')
            examples.append(Example(terms))


if __name__ == '__main__':
    concrete_train_file = "Data/concrete/train.csv"
    concrete_test_file = "Data/concrete/test.csv"





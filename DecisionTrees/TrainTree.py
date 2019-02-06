



class example:
    def __init__(self, values):
        self.attributes = values[0: len(values)-1]
        self.label = values[len(values)-1]


class decision_tree:
    def __init__(self):
        self.decision_attribute = None
        self.decision = dict()



def train_from_examples(examples):


    return decision_tree


def examples_from_attributes(file):
    examples = list()
    with open(file, 'r') as train_data:
        for line in train_data:
            terms = line.strip().split(',')
            examp = example(terms)
            examples.append(examp)

    return examples


def tree_from_file(file):
    return decision_tree
    

def main():
    command = input('Am I learning today, or predicting?')
    command = command.lower();

    if command == "learning" or command == "train":
        file = input("Training data file: ")
        examples = examples_from_attributes(file)
        tree = train_from_examples(examples)

    elif command == "predict":
        file = input("Using what trained model file?")
        tree = tree_from_file(file)


main();
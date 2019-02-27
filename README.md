# MachineLearningStudy
This is a machine learning library developed by Jennifer Nelson for CS5350/6350 at the University of Utah

### Decision Trees
Decision trees are encapsulated in the DecisionTree class. Instantiating a DecisionTree object requires training data and several parameters (including the metric used to calculate gain.)

`tree = DecisionTree(max_depth, depth, examples, gain_metric, most_common_label, is_categoric_attribute, random_subset=False, random_subset_size=0)`

`examples` should a list of `Example` objects. These objects are easily created from a .csv file using the `examples_from_file` method. 
`gain_metric` is a function for computing the information in an example subset. It has the signature `gain_metric(examples) -> float`
`most_common_label` is the most common label in the above tree, used for the recursive structure of DecisionTree. This can safely be `None` if examples is non-empty.
`is_categoric_attribute` is a list of the same length as the examples' attribute size. Each entry indicates whether or not that attribute index should be treated as numeric or categoric. `examples_from_file` will return this.
`random_subset` and `random_subset_size` are used to make the decision tree a randomized tree. `random_subset_size` indicates how many attributes may be considered for splitting upon.

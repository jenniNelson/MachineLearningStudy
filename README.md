# MachineLearningStudy
This is a machine learning library developed by Jennifer Nelson for CS5350/6350 at the University of Utah



### Running
The run.sh file will start several testing processes which can take several hours to complete. It will print out where the data can be found.

I don't suggest running it unless you have to.



### Decision Trees
Decision trees are encapsulated in the DecisionTree class. Instantiating a DecisionTree object requires training data and several parameters (including the metric used to calculate gain.)

`tree = DecisionTree(max_depth, depth, examples, gain_metric, most_common_label, categorics, random_subset=False, random_subset_size=0)`

`examples` should a list of `Example` objects. These objects are easily created from a .csv file using the `examples_from_file` method. 

`gain_metric` is a function for computing the information in an example subset. It has the signature `gain_metric(examples) -> float`

`most_common_label` is the most common label in the above tree, used for the recursive structure of DecisionTree. This can safely be `None` if examples is non-empty.

`categorics` is a list of the same length as the examples' attribute size. Each entry indicates whether or not that attribute index should be treated as numeric or categoric. `examples_from_file` will return this.

`random_subset` and `random_subset_size` are used to make the decision tree a randomized tree. `random_subset_size` indicates how many attributes may be considered for splitting upon.

### Adaboost
 `adaBoost_vote_weights_and_stumps(examples, num_stumps, categorics)`
 
 This method will create two lists of the same size, one of decision tree stumps (or longer, if a stump is worse than chance) and one of each decision stump's corrseponding vote weight.
 
 (Note that `categorics` is a list of which attributes are numeric and which categoric (see above).)
 
 In order to run, pass these two lists and the desired number of trees to use in a decision into the following method:
 `adaDecide(sample, vote_weights, stumps, num_to_consider)`

### Bagging and Random Forests
`baggy_trees(examples, subset_size, num_trees, categorics)`

This method returns a list of trees of the specified size, all trained on a random subset (`subset_size`) of the examples.

`baggy_forest(examples, attribute_subset_size, example_subset_size, num_trees, categorics)`

This method performs similarly to baggy_trees, but returns a list of trees trained using both a random subset of examples *and* a random subset of attributes to choose from.

To make a prediction with either of these, pass the tree list into 

`bagging_decision(sample, trees, num_trees_to_use)`

### LMS: Batch and Stochastic Gradient Regression

   > Note that examples here is created using a different function than that of DecisionTrees and ensemble methods. This is to ensure all examples are numeric and augmented with an additional 1, for the b in a weight vector.
   
   > This method is `examples_from_file_with_b(filename)`

`lms_gradient_descent(examples, learning_rate, threshold, max_iterations)` and

`stochastic_descent(examples, learning_rate, threshold, max_iterations)`

 both return `(weights, iteration, error)`, where `weights` is the learned weight vector, `iteration` is how many iterations the training took, and `error` is the final least-mean-square error of the weight vector. (As computed by `lms_error(examples, weights)`)

 To predict with the returned weight vector, call `predict(sample, weights)`.
 
 Several other functions for computing average error, gradient, and similar also exist.

 
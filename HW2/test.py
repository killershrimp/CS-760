import decision_tree
import util

inp_f = "data/D2.txt"

data = util.read_data(inp_f)
train, test = util.split_data(data, 0.8192)

# splits = decision_tree.determine_candidate_splits(train)
# for i in range(len(splits)):
#     split = splits[i]
#     print("\\item", str(int(split[0])) + "th feature;",
#           ("high" if split[2] else "low") + "er than", str(split[1])
#           + "; gain ratio:", gain_ratio(data, split))

DecisionTree = decision_tree.make_subtree(train)

accuracy = decision_tree.test_subtree(test, DecisionTree)
print("Accuracy:", accuracy)
print("done training")

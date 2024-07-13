# Report on Decision Tree based ID3 Algorithm

## Overview

The ID3 (Iterative Dichotomiser 3) algorithm is a foundational technique in the realm of decision trees, utilized for constructing a decision tree from a dataset. It operates on a recursive, top-down approach to partition the dataset into subsets, selecting the most informative attributes at each step based on information theory, specifically entropy.

## Structure of a Decision Tree

A decision tree consists of several components:

- **Root Node:** The starting point of the tree, representing the entire dataset.
- **Internal Nodes:** Nodes within the tree where data is split based on specific attributes.
- **Leaf Nodes:** The terminal points of the tree, representing the final classification or decision.

The tree structure visually resembles an inverted tree, starting from the root and branching out to the leaves, with each decision point representing a split based on a feature.

## Basic Terminologies

- **Root Node:** Represents the entire dataset and is the topmost node.
- **Internal Node:** A node with child nodes, representing a decision point.
- **Leaf Node:** A node with no children, representing a final classification or decision.
- **Splitting:** Dividing the dataset into subsets based on a feature.
- **Splitting Criterion:** The feature and value used for splitting the data.
- **Entropy:** A measure of impurity or disorder in the dataset.
- **Information Gain:** The reduction in entropy after a dataset is split based on a feature.
- **Pruning:** Reducing the size of the decision tree by removing parts that do not improve prediction accuracy.
- **Maximum Depth:** The maximum number of levels from the root to the leaf.
- **Feature Importance:** The relevance of each feature in making predictions.

## ID3 Algorithm

The ID3 algorithm aims to create a decision tree that partitions the dataset into subsets as homogeneously as possible. The steps involved in the algorithm are:

1. **Calculate Entropy:** Measure the disorder or impurity in the dataset.
2. **Calculate Information Gain:** Determine the reduction in entropy for each feature.
3. **Select the Best Feature:** Choose the feature with the highest information gain.
4. **Split the Dataset:** Divide the dataset based on the selected feature.
5. **Repeat Recursively:** Apply the above steps to each subset until all instances are classified or no further splits can be made.

### Dataset Example

For illustration, let's consider a dataset related to surfing conditions:

| Day  | Outlook  | Temp | Humidity | Wind   | Surfing? |
|------|----------|------|----------|--------|----------|
| D1   | Sunny    | Hot  | High     | Weak   | No       |
| D2   | Sunny    | Hot  | High     | Strong | No       |
| D3   | Overcast | Hot  | High     | Weak   | Yes      |
| D4   | Rain     | Mild | High     | Weak   | Yes      |
| D5   | Rain     | Cool | Normal   | Weak   | Yes      |
| D6   | Rain     | Cool | Normal   | Strong | No       |
| D7   | Overcast | Cool | Normal   | Weak   | Yes      |
| D8   | Sunny    | Mild | High     | Weak   | No       |
| D9   | Sunny    | Cold | Normal   | Weak   | Yes      |
| D10  | Rain     | Mild | Normal   | Strong | Yes      |
| D11  | Sunny    | Mild | Normal   | Strong | Yes      |
| D12  | Overcast | Mild | High     | Strong | Yes      |
| D13  | Overcast | Hot  | Normal   | Weak   | Yes      |
| D14  | Rain     | Mild | High     | Strong | No       |

### Step-by-Step Implementation

#### Step 1: Calculate Total Entropy

First, we calculate the entropy of the entire dataset.

\[
H(S) = -p(Yes) \log_2(p(Yes)) - p(No) \log_2(p(No))
\]

Where:
- \( p(Yes) = \frac{9}{14} \)
- \( p(No) = \frac{5}{14} \)

\[
H(S) = -\left(\frac{9}{14}\right) \log_2\left(\frac{9}{14}\right) - \left(\frac{5}{14}\right) \log_2\left(\frac{5}{14}\right) = 0.94
\]

#### Step 2: Calculate Entropy for Each Feature

We calculate the entropy for each feature (Outlook, Temperature, Humidity, Wind) and their respective values.

For example, for the feature "Outlook":

\[
H(\text{Outlook} = \text{Sunny}) = -\left(\frac{2}{5}\right) \log_2\left(\frac{2}{5}\right) - \left(\frac{3}{5}\right) \log_2\left(\frac{3}{5}\right) = 0.971
\]

We perform similar calculations for "Rain" and "Overcast."

#### Step 3: Calculate Information Gain

Information Gain for a feature:

\[
\text{Info Gain}(\text{Feature}) = H(S) - \left(\sum \left(\frac{\text{subset size}}{\text{total size}} \times H(\text{subset})\right)\right)
\]

We calculate this for each feature and select the one with the highest information gain.

#### Step 4: Build the Decision Tree

Using the feature with the highest information gain, we create the root node and split the dataset accordingly. This process is repeated recursively for each subset until all instances are classified.

### Python Implementation

```python
import pandas as pd
import numpy as np

# Step 1: Calculate Total Entropy
def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_entr = 0
    for c in class_list:
        total_class_count = train_data[train_data[label] == c].shape[0]
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row)
        total_entr += total_class_entr
    return total_entr

# Step 2: Calculate Entropy for Each Feature
def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count
            entropy_class = - probability_class * np.log2(probability_class)
        entropy += entropy_class
    return entropy

# Step 3: Calculate Information Gain
def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy
    return calc_total_entropy(train_data, label, class_list) - feature_info

# Step 4: Find the Most Informative Feature
def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)
    max_info_gain = -1
    max_info_feature = None
    for feature in feature_list:
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature
    return max_info_feature

# Step 5: Generate Sub-Tree
def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {}
    for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        assigned_to_node = False
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]
            if class_count == count:
                tree[feature_value] = c
                train_data = train_data[train_data[feature_name] != feature_value]
                assigned_to_node = True
        if not assigned_to_node:
            tree[feature_value] = "?"
    return tree, train_data

# Step 6: Build the Decision Tree
def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:
        max_info_feature = find_most_informative_feature(train_data, label, class_list)
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list)
        next_root = None
        if prev_feature_value != None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:
            root[max_info_feature] = tree
            next_root

 = root[max_info_feature]
        for node, branch in list(next_root.items()):
            if branch == "?":
                feature_value_data = train_data[train_data[max_info_feature] == node]
                make_tree(next_root, node, feature_value_data, label, class_list)

# Training Data
dataset = {
    'Day': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14'],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Surfing?': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

train_data = pd.DataFrame(dataset)
label = 'Surfing?'
class_list = train_data[label].unique()
tree = {}
make_tree(tree, None, train_data, label, class_list)
print(tree)
```

### Example of a Generated Decision Tree

Based on the calculations and splitting criteria, a possible decision tree for the given dataset might look like this:

```
Outlook
|   Rain
|   |   Wind
|   |   |   Weak: Yes
|   |   |   Strong: No
|   Overcast: Yes
|   Sunny
|   |   Humidity
|   |   |   High: No
|   |   |   Normal: Yes
```

## Conclusion

The ID3 algorithm is a powerful tool for constructing decision trees, which can be used for various classification tasks. By leveraging entropy and information gain, it ensures that the most informative features are selected at each step, leading to a robust and interpretable model. This report outlines the theoretical foundation and provides a practical implementation example, illustrating the process of building a decision tree using ID3.


# Decision Tree Regression Model

## Description

This Python code imports the necessary libraries, reads the dataset, trains a Decision Tree Regression model, prints the feature importance, tree depth, and stopping criteria, and plots the decision tree.

## Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
```

## Dataset

```python
dataset = pd.read_csv('students.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
```

## Training the Model

```python
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
```

## Feature Importance

```python
feature_importance = regressor.feature_importances_
print("Feature Importance:", feature_importance)
```

## Tree Depth

```python
tree_depth = regressor.get_depth()
print("Tree Depth:", tree_depth)
```

## Tree plot



## Stopping Criteria

```python
max_tree_depth = regressor.get_params()['max_depth']
min_samples_split = regressor.get_params()['min_samples_split']
min_samples_leaf = regressor.get_params()['min_samples_leaf']
print("Max Tree Depth (Stopping Criteria):", max_tree_depth)
print("Min Samples Split (Stopping Criteria):", min_samples_split)
print("Min Samples Leaf (Stopping Criteria):", min_samples_leaf)
```

## Plotting the Decision Tree

```python
plt.figure(figsize=(15, 10))
plot_tree(regressor, filled=True, rounded=True, fontsize=10)
plt.show()
```

## Usage

To use this code, simply save it as a Python file (e.g. `decision_tree_regression.py`) and run it in a terminal or command prompt. The code will train a Decision Tree Regression model on the `students.csv` dataset and print the feature importance, tree depth, and stopping criteria. The code will also plot the decision tree.

## Requirements

* Python 3
* NumPy
* Matplotlib
* Pandas
* scikit-learn

## Example Output

```
Predicted Final Exam Score: [91.]
```

## Conclusion

This Python code provides a simple example of how to train and use a Decision Tree Regression model.


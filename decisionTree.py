import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('students.csv')
X = data.drop(['Roll No','Final Exam Score'],axis=1)
y = data['Final Exam Score']
X_encoded = pd.get_dummies(X, columns=['Extracurricular', 'School Type', 'Parental Education', 'Gender'], drop_first=True)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_encoded,y)
feature_order = X_encoded.columns.tolist()

new_student_data = pd.DataFrame({
    'Study Hours/Week': [10],
    'Previous Scores': [85],
    'Family Income ($)': [45000],
    'Distance to School (miles)': [2.5],
    'Number of Friends': [3],
    'Commute Time (minutes)': [20],
    'Extracurricular_Yes': [1],
    'School Type_Public': [1],
    'Parental Education_Master\'s': [0],
    'Gender_Male': [1]
}, columns=feature_order)  # Ensure the order matches
predicted_score = regressor.predict(new_student_data)
print("Predicted Final Exam Score:", predicted_score)

#Understanding the decision tree

feature_importance = regressor.feature_importances_
print("Feature Importance:", feature_importance)

tree_depth = regressor.get_depth()
print("Tree Depth:", tree_depth)

max_tree_depth = regressor.max_depth
min_samples_split = regressor.min_samples_split
min_samples_leaf = regressor.min_samples_leaf

print("Max Tree Depth (Stopping Criteria):", max_tree_depth)
print("Min Samples Split (Stopping Criteria):", min_samples_split)
print("Min Samples Leaf (Stopping Criteria):", min_samples_leaf)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Plot the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(regressor, filled=True, rounded=True, fontsize=10)
plt.show()

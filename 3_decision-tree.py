# load in the relevant packages
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree

# set file paths
cwd = os.getcwd()
project = Path(f'{cwd}/Assignments/Project')
output = Path(f'{project}/graphs-charts')

dataset = pd.read_csv(f'{project}/data.csv')

# let's do some recoding
dataset['sexuality'].replace(['Identifies as strictly heterosexual', 'Does not identify as strictly heterosexually desiring'],
                        [0, 1], inplace=True)

dataset['insured'].replace(['Uninsured', 'Insured'],
                        [0, 1], inplace=True)

# let's recode sex
dataset['sex'].replace(['1 Male', '2 Female'],
                        [0, 1], inplace=True)

# let's recode race
dataset['race'].replace(['Non-Hispanic White', 'Non-Hispanic Black', 'Other non-Hispanic', 'Hispanic'],
                        [0, 1, 2, 3], inplace=True)

# let's recode age
dataset['age'].replace(['65+', '18-24', '25-34', '35-44', '45-54', '55-64'],
                        [0, 1, 2, 3, 4, 5], inplace=True)

# let's recode region
dataset['region'].replace(['Northeast', 'Midwest', 'West', 'South'],
                        [0, 1, 2, 3], inplace=True)

inputs = dataset[['insured', 'sexuality', 'race', 'age', 'region', 'sex']]

# let's run the model now
X = inputs.loc[:, inputs.columns != 'insured']
y = inputs.loc[:, inputs.columns == 'insured']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

plt.figure(figsize=(12,8))
tree.plot_tree(tree_model.fit(X_test, y_train))
plt.show()

score = accuracy_score(y_test, y_pred)

plt.figure(figsize=(12,8))
tree.plot_tree(tree_model.fit(X_train, y_train))
plt.show()

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15);
plt.savefig(f'{output}/decision-tree-confusion-matrix')
plt.show()

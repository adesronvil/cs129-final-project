# load in the relevant packages
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

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
logit = LogisticRegression(solver='liblinear', multi_class='ovr')
kfold = StratifiedKFold(n_splits=4, random_state=1, shuffle=True)
cv_results = cross_val_score(logit, X_train, y_train, cv=kfold, scoring='accuracy')

logit.fit(X_train, y_train.values.ravel())

y_pred = logit.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logit.score(X_test, y_test)))

classification_report = classification_report(y_test, y_pred, output_dict=True)
df_classification_report = pd.DataFrame(classification_report).transpose()
df_classification_report.to_csv(f'{output}expanded-classification-report.csv')

print(classification_report)

logit_roc_auc = roc_auc_score(y_test, logit.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logit.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(f'{output}/expanded-Log_ROC.png')
plt.show()

score = logit.score(X_test, y_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15);
plt.savefig(f'{output}/expanded-confusion-matrix')
plt.show()

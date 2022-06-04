# load in the relevant packages
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# set file paths
cwd = os.getcwd()
project = Path(f'{cwd}/Assignments/Project')
output = Path(f'{project}/graphs-charts')

dataset = pd.read_csv(f'{project}/nhis.csv')

# let's check for missing values
dataset.isnull().sum()

# let's get info on the data
dataset.info()

# let's convert floats into integers, but not income yet
float_vars = ['srvy_yr', 'kids']

for var in float_vars:
    dataset[var] = dataset[var].apply(np.int64)

# let's get information
dataset.describe()
dataset.describe(include=object)

# let's drop 'wrk_no' variable
dataset = dataset.drop(columns=['wrk_no'])

# create dictionary for name
new_names = {'srvy_yr': 'year', 'relation_grp': 'marriage', 'hlthstat_grp': 'health-status', 'reg_grp': 'region', 'healt_insured': 'insurance', 'smkoq_grp': 'smoking', 'educati_grp': 'education', 'rac_grp': 'race', 'age_grp1': 'age'}

dataset = dataset.rename(columns=new_names)

dataset.to_csv(f'{project}/data.csv', index=False)

# let's visualize our missing data
missing_heatmap = sns.heatmap(dataset.isnull())
missing_heatmap.figure.tight_layout()
missing_heatmap.set_title('Heatmap of NHIS, 2013-2018')
plt.show()
fig = missing_heatmap.get_figure()
fig.savefig(f'{output}/missing-heatmap.png')

# let's check in on our outcome variable
insured_countplot = sns.countplot(x='insured', data=dataset)
insured_countplot.set_title('Insured and uninsured from NHIS, 2013-2018')
insured_countplot.figure.tight_layout()
plt.show()
fig = insured_countplot.get_figure()
fig.savefig(f'{output}/insured-countplot.png')

# let's do some recoding
dataset['sexuality'].replace(['Identifies as strictly heterosexual', 'Does not identify as strictly heterosexually desiring'],
                        [0, 1], inplace=True)

dataset['insured'].replace(['Uninsured', 'Insured'],
                        [0, 1], inplace=True)

# let's build our first model
X = dataset.loc[:, dataset.columns == 'sexuality']
y = dataset.loc[:, dataset.columns == 'insured']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)
logistic = LogisticRegression(solver='liblinear', multi_class='ovr')
model = logistic.fit(X_train, y_train.values.ravel())

y_pred = logistic.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logistic.score(X_test, y_test)))

classification_report = classification_report(y_test, y_pred, output_dict=True)
df_classification_report = pd.DataFrame(classification_report).transpose()
df_classification_report.to_csv(f'{output}simple-classification-report.csv')

print(classification_report)

logistic_roc_auc = roc_auc_score(y_test, logistic.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logistic.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logistic_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(f'{output}/simple-Log_ROC.png')
plt.show()

score = logistic.score(X_test, y_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15);
plt.savefig(f'{output}/simple-confusion-matrix')
plt.show()
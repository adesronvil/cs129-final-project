# load in the relevant packages
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# set file paths
data_path = Path('/Users/allexdesronvil/Library/CloudStorage/Box-Box/Projects/LGB Health Insurance')
cwd = os.getcwd()
project = Path(f'{cwd}/Assignments/Project')
output = Path(f'{project}/graphs-charts')

if not os.path.exists(output):
    os.mkdir(output)

# load in dataset
df = pd.read_stata(f'{data_path}/nhis_clean_income.dta')

# select requisite variables
data = df[['srvy_yr', 'lgb', 'sexuality', 'insured', 'relation_grp', 'hlthstat_grp', 'reg_grp', 'healt_insured', 'smkoq_grp', 'educati_grp', 'rac_grp', 'age_grp1', 'wrk_no', 'kids', 'income', 'sex']]

# export to csv
data.to_csv(f'{project}/nhis.csv', index=False)
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

df = pd.read_csv('nhanes_metSyn_20220316.csv')
print('Sample w/o label', [i for i, x in enumerate(df['allcausedeath'].isna().values) if x ==True])
df = df.fillna(0)
y = df['allcausedeath'].values
df = df.drop(['SEQN', 'followupmonth', 'allcausedeath'], axis=1)
print('Data shape', df.shape)
chi_scores = chi2(df,y)
p_values = pd.Series(chi_scores[1],index = df.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()
plt.show()

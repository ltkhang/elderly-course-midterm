import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import preprocessing


df = pd.read_csv('nhanes_metSyn_20220316.csv')
print('Sample w/o label', [i for i, x in enumerate(df['allcausedeath'].isna().values) if x ==True])
df = df.fillna(0)
y = df['allcausedeath'].values
df = df.drop(['SEQN', 'followupmonth', 'allcausedeath'], axis=1)
print('Data shape', df.shape)
best_mean = 0
best_std = 0
best_num_feat = 0
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

df = preprocessing.MinMaxScaler().fit_transform(df)
for i in range(31):
    num_feat = 12 + i
    # select top feature with Chi-Square test
    X = SelectKBest(chi2, k=num_feat).fit_transform(df, y)
    scores = cross_val_score(LogisticRegression(random_state=1234), X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    mean = np.mean(scores)
    std = np.std(scores)
    if mean > best_mean:
        best_mean = mean
        best_std = std
        best_num_feat = num_feat
print('Best {:.2f} +- {:.2f} with number of feature {}'.format(best_mean * 100, best_std * 100, best_num_feat))


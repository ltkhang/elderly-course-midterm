import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('nhanes_metSyn_20220316.csv')
print('Sample w/o label', [i for i, x in enumerate(df['allcausedeath'].isna().values) if x ==True])
df = df.fillna(0)
y = df['allcausedeath'].values
df = df.drop(['SEQN', 'followupmonth', 'allcausedeath'], axis=1)
print('Data shape', df.shape)
df = preprocessing.MinMaxScaler().fit_transform(df)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

principalDf['target'] = y
finalDf = principalDf

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
target_labels = ['non-allcausedeath', 'allcausedeath']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(target_labels)
ax.grid()
plt.show()
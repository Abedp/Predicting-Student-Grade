import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import kstest, shapiro
from scipy.stats import spearmanr
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



df = pd.read_csv('data.csv')

sb.heatmap(df.corr(method='spearman'), annot=True)
df.describe()

df100 = df.head(100)
pysics = df100['Pysics']
science = df100['Science']
statistic = df100['Statistics']
math = df100['Math']

fig, ax = plt.subplot()

df.describe()

plt.plot(statistic, math)
sb.scatterplot(df['Statistics'], df['Math'])
sb.pairplot(df)

ksdata = kstest(df['Statistics'], 'norm')

c, p = spearmanr(df['Science'],df['Math'])
df.corr(method='spearman')

plt.boxplot(df['Statistics'])

def plotbox(df, ft):
    df.boxplot(column=[ft])
    plt.grid(False)
    plt.show()

plotbox(df, 'Pysics')

def list_outliers(df, ft):
    q1 = df[ft].quantile(0.25)
    q3 = df[ft].quantile(0.75)
    IQR = q3-q1
    upper = q3 + 1.5 * IQR
    lower = q1 - 1.5 * IQR
    
    outlier = df.index[ (df[ft] < lower) | (df[ft] > upper)]
    return outlier

outlier_index = []

for feature in ['Pysics', 'Science', 'Statistics', 'Math']:
    outlier_index.extend(list_outliers(df, feature))


def remove(df, routlier):
    routlier = sorted(set(routlier))
    df = df.drop(routlier)
    return df

df1 = remove(df,outlier_index)

sb.barplot(x = 'Science', y='Math', data=df1)
sb.pairplot(df1)

plt.boxplot(df1['Math'])
sb.scatterplot(df1['Pysics'], df1['Math'])

ksdata = kstest(df1['Math'], 'norm')
swdata = shapiro(df['Statistics'])

df1.corr(method='spearman')

X = df1.iloc[:,0:3].values
Y = df1.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.25)

svr_model = SVR(kernel='rbf', C=0.1, epsilon=0.1)
svr_model.fit(X_train,y_train)
y_pred = svr_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))


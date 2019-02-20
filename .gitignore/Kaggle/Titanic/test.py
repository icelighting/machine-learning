from sys import argv
import sys
from scipy import stats
import pandas as pd
import seaborn as sns

df = pd.read_csv(r'D:\data analysis\Kaggle\Titanic\input\train.csv')
print(df.head())
means, std = stats.norm.fit(df['Parch'])
print('Means',means)


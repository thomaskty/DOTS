
import pandas as pd

df = pd.read_csv('./machine_learning/data/blackfriday.csv')
df.head()
df.shape
df.columns

df['Occupation'].value_counts()

# Group by the 'category' column and apply sorting to each group
sorted_groups = df.groupby('Occupation').apply(lambda x : x.sort_values('Age',ascending = False).iloc[1])
sorted_groups[['Age']]


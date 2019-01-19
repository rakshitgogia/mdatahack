import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

df = pd.read_csv('data.csv')

# remove 'FH' and convert to int
df['ID'] = df['Player Name'].str[2:].astype(int)

# drop Player Name
df.drop('Player Name', axis=1, inplace=True)

# drop useless features
df.drop(['Period Number', 'Date_obj', 'Start Time', 'End Time', 'Day Name', 'Month Name'], axis=1, inplace=True)

# print(df['ID'].value_counts())

# relabel target
df.loc[df['Position Name'] == 'Goal Keeper', 'Position Name'] = 'Goalkeeper'
df.loc[df['Position Name'] == 'Forward/Mid', 'Position Name'] = 'Mid/Forward'
df.loc[df['Position Name'] == 'Midfield', 'Position Name'] = 'Mid'

# convert Activity Name to practice (0) or game (1)
df['practice_or_game'] = df['Activity Name'].str.contains('Practice', case=False) == True
df.loc[df['practice_or_game'] == False, 'practice_or_game'] = 0
df.loc[df['practice_or_game'] == True, 'practice_or_game'] = 1

# print(df['practice_or_game'].value_counts())

df.drop('Activity Name', axis=1, inplace=True)

# clean Period Name
df= df[(df['Period Name'] != "Water Break")]

# remove misleading practices
df = df[(df["Period Name"]).str.contains("OT practice", case=False) == False]
df = df[(df["Period Name"]).str.contains("11 v 11", case=False) == False]
df = df[(df["Period Name"]).str.contains("11v11", case=False) == False]
df = df[(df["Period Name"]).str.contains("9 v 9", case=False) == False]
df = df[(df["Period Name"]).str.contains("9v9", case=False) == False]

# combine all warmups
df.loc[df['Period Name'].str.contains('war', case=False), 'Period Name'] = 'Warmup'

# combine all first and session into GAME
df.loc[df['Period Name'].str.contains('firs', case=False), 'Period Name'] = 'Game'
df.loc[df['Period Name'].str.contains('secon', case=False), 'Period Name'] = 'Game'

# label encode each Period Name
# Warmup (0), Game (1), Session (2), Misc. (3)
df.loc[df['Period Name'] == 'Warmup', 'new_period'] = 0
df.loc[df['Period Name'] == 'Game', 'new_period'] = 1
df.loc[df['Period Name'] == 'Session', 'new_period'] = 2
df['new_period'] = df['new_period'].fillna(value=3)

df.drop('Period Name', axis=1, inplace=True)

# print(df['new_period'].value_counts())

# encode target
dic = {'Back': 0, 'Forward': 1, 'Mid/Back': 2,
       'Mid': 3, 'Mid/Forward': 4, 'Goalkeeper': 5}

df["Position Name"] = df["Position Name"].apply(lambda x: dic[x])

df["ID"] = df["ID"].astype('int8')
df["practice_or_game"] = df["practice_or_game"].astype('int8')
df["new_period"] = df["new_period"].astype('int8')

print(df.dtypes)

# print(df['Position Name'].value_counts())


# #filter out water breaks
# df_cleaned = df[(df["Activity Name"] != "Water Break")]
#
# #filter out invalid distances
# df_cleaned = df_cleaned[(df_cleaned["Total Distance"] < 20000) & (df_cleaned["Total Distance"] > 50)]
#
# #filter out bad player loads
# df_cleaned = df_cleaned[(df_cleaned["Total Player Load"] < 1250) & (df_cleaned["Total Player Load"] > 50)]
#
# #filter out bad 'Total IMAs'
# df_cleaned = df_cleaned[(df_cleaned["Tot IMA"] < 650) & (df_cleaned["Tot IMA"] > 20)]
#
# #filter out bad high speed distance
# df_cleaned = df_cleaned[(df_cleaned["High Speed Distance"] < 1000)]
#
# #filter out bad max velocities
# df_cleaned = df_cleaned[(df_cleaned["Maximum Velocity"] > 2) & (df_cleaned["Maximum Velocity"] < 8)]
#

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

print(df['ID'].value_counts())

# relabel target
df.loc[df['Position Name'] == 'Goal Keeper', 'Position Name'] = 'Goalkeeper'
df.loc[df['Position Name'] == 'Forward/Mid', 'Position Name'] = 'Mid/Forward'
df.loc[df['Position Name'] == 'Midfield', 'Position Name'] = 'Mid'

# convert Activity Name to practice (0) or game (1)
df['practice_or_game'] = df['Activity Name'].str.contains('Practice', case=False) == True
df['practice_or_game'][df['practice_or_game'] == False] = '0'
df['practice_or_game'][df['practice_or_game'] == True] = '1'

df.drop('Activity Name', axis=1, inplace=True)

# clean Period Name
df= df[(df['Period Name'] != "Water Break")]

# Remove misleading practices
df = df[(df["Period Name"]).str.contains("OT practice", case=False) == False]
df = df[(df["Period Name"]).str.contains("11 v 11", case=False) == False]
df = df[(df["Period Name"]).str.contains("11v11", case=False) == False]
df = df[(df["Period Name"]).str.contains("9 v 9", case=False) == False]
df = df[(df["Period Name"]).str.contains("9v9", case=False) == False]

print(df['Period Name'].value_counts())

# Encode features
dic = {'Back': 0, 'Forward': 1, 'Mid/Back': 2,
       'Mid': 3, 'Mid/Forward': 4, 'Goalkeeper': 5}

df["Position Name"] = df["Position Name"].apply(lambda x: dic[x])

print(df['Position Name'].value_counts())


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

import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm_notebook as tqdm

import warnings
warnings.simplefilter(action='ignore')

TARGET = 'Position Name'

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

# remove duplicated Session
df = df[(df["Period Name"]).str.contains("Session", case=False) == False]

# combine all warmups
df.loc[df['Period Name'].str.contains('war', case=False), 'Period Name'] = 'Warmup'

# combine all first and session into GAME
df.loc[df['Period Name'].str.contains('firs', case=False), 'Period Name'] = 'Game'
df.loc[df['Period Name'].str.contains('secon', case=False), 'Period Name'] = 'Game'

# label encode each Period Name
# Warmup (0), Game (1), Session (2), Misc. (3)
df.loc[df['Period Name'] == 'Warmup', 'period_name'] = 0
df.loc[df['Period Name'] == 'Game', 'period_name'] = 1
df['period_name'] = df['period_name'].fillna(value=2)

df.drop('Period Name', axis=1, inplace=True)

# print(df['period_name'].value_counts())

# encode target
dic = {'Back': 0, 'Forward': 1, 'Mid/Back': 2,
       'Mid': 3, 'Mid/Forward': 4, 'Goalkeeper': 5}
df["Position Name"] = df["Position Name"].apply(lambda x: dic[x])

# convert new features to int8
df["ID"] = df["ID"].astype('int8')
df["practice_or_game"] = df["practice_or_game"].astype('int8')
df["period_name"] = df["period_name"].astype('int8')

# filter out invalid distances
df = df[(df["Total Distance"] < 20000) & (df["Total Distance"] > 50)]

# filter out bad player loads
df = df[(df["Total Player Load"] < 1250) & (df["Total Player Load"] > 50)]

# filter out bad 'Total IMAs'
df = df[(df["Tot IMA"] < 650) & (df["Tot IMA"] > 20)]

# filter out bad high speed distance
df = df[(df["High Speed Distance"] < 1000)]

# filter out bad IMA Accel Low
df = df[(df["IMA Accel Low"] < 30)]

# filter out bad IMA Accel Medium
df = df[(df["IMA Accel Medium"] < 11)]

# filter out bad IMA Accel High
df = df[(df["IMA Accel High"] < 13.1)]

# filter out bad IMA Decel Low
df = df[(df["IMA Decel Low"] < 31)]

# filter out bad IMA Decel Medium
df = df[(df["IMA Decel Medium"] < 15)]

# filter out bad IMA Decel High
df = df[(df["IMA Decel High"] < 10)]

# filter out bad Band 1
df = df[(df["Velocity Band 1 Total Distance"] < 2200)]

# filter out bad Band 2
df = df[(df["Velocity Band 2 Total Distance"] < 4000)]

# filter out bad Band 4
df = df[(df["Velocity Band 4 Total Distance"] < 6200)]

# filter out bad Band 5
df = df[(df["Velocity Band 5 Total Distance"] < 500)]

# filter out bad max velocities
df = df[(df["Maximum Velocity"] > 2) & (df["Maximum Velocity"] < 8)]

# removed all null
df = df.dropna(how='any',axis=0)

features_n = ['Total Player Load', 'Total Distance', 'Tot IMA',
              'High Speed Distance', 'IMA Accel High', 'IMA Accel Low',
              'IMA Accel Medium', 'IMA CoD Left High', 'IMA CoD Left Low',
              'IMA CoD Left Medium', 'IMA CoD Right High', 'IMA CoD Right Low',
              'IMA CoD Right Medium', 'IMA Decel High', 'IMA Decel Low',
              'IMA Decel Medium', 'Maximum Velocity', 'Meterage Per Minute',
              'Player Load Per Minute', 'Velocity Band 1 Total Distance',
              'Velocity Band 2 Total Distance', 'Velocity Band 3 Total Distance',
              'Velocity Band 4 Total Distance', 'Velocity Band 5 Total Distance']

# normalize data by players
def normalize_by_players(df):
    players = df["ID"].unique()
    for player in players:
        ids = df[df["ID"] == player].index
        scaler = MinMaxScaler()
        df.loc[ids, features_n] = scaler.fit_transform(df.loc[ids, features_n])

    return df

# # heatmap
# corrmat = df.corr()
# k = 10 # number of variables for heatmap
# cols = corrmat.nlargest(k, 'Total Player Load')['Total Player Load'].index
# cm = np.corrcoef(df[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

# # violinplot
# plt.figure(figsize=(8,6))
# sns.violinplot(x='Position Name', y='Total Player Load', data=df)
# plt.ylabel("Player Load", fontsize=12)
# plt.xlabel("Position Name", fontsize=12)
# plt.title("Position Name vs Player Load", fontsize=15)
# plt.show()

# # each position vs count
# plt.figure(figsize=(8,6))
# sns.countplot(df[TARGET])
# plt.xlabel("Position Code", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.title("Position vs Count", fontsize=15)
# plt.show()

# # ?
# plt.figure(figsize=(8,6))
# sns.catplot(x=df[TARGET], y=df['ID'], data=df)
# plt.xlabel("Position Code", fontsize=12)
# plt.ylabel("Num of Unique Players", fontsize=12)
# plt.title("Position Code vs Num of Unique Players", fontsize=15)
# plt.show()

# normalize_by_players(df)

# dist for Total Distance
# plt.figure(figsize=(8,6))
# # sns.distplot(df['Total Distance'], label='Distance')
# sns.distplot(np.log(df['Total Distance']), label='Log Distance')
# plt.legend()
# plt.xlabel("Total Distance", fontsize=12)
# plt.title("Total Distance Distribution", fontsize=15)
# plt.show()

# # dist for Tot IMA
# plt.figure(figsize=(8,6))
# sns.distplot(np.log(df['Tot IMA']), label='IMA')
# plt.legend()
# plt.xlabel("Tot IMA", fontsize=12)
# plt.title("Tot IMA Distribution", fontsize=15)
# plt.show()

def des_series(df_feature):
    print('summary: {}'.format(df_feature.describe()))
    print('.9 quantile: {}'.format(df_feature.quantile(.9)))
    print('.99 quantile: {}'.format(df_feature.quantile(.99)))
    print('.995 quantile: {}'.format(df_feature.quantile(.995)))
    print('.999 quantile: {}'.format(df_feature.quantile(.999)))

# des_series(df['Velocity Band 3 Total Distance'])

print(pd.cut(df['Velocity Band 1 Total Distance'], 5))

# # dist for IMA Accel
# plt.figure(figsize=(8,6))
# sns.distplot(df['IMA Accel High'], label='H')
# sns.distplot(df['IMA Accel Low'], label='L')
# sns.distplot(df['IMA Accel Medium'], label='M')
# plt.legend()
# plt.xlabel("IMA Accel ", fontsize=12)
# plt.title("IMA Accel Distribution", fontsize=15)
# plt.show()

# # dist for IMA Decel
# plt.figure(figsize=(8,6))
# sns.distplot(df['IMA Decel High'], label='H')
# sns.distplot(df['IMA Decel Low'], label='L')
# sns.distplot(df['IMA Decel Medium'], label='M')
# plt.legend()
# plt.xlabel("IMA Decel", fontsize=12)
# plt.title("IMA Decel Distribution", fontsize=15)
# plt.show()

print(df.shape)
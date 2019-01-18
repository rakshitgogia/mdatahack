# -*- coding: utf-8 -*-
"""Starter Code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PeJFubXx2phbPKlQvg0Bd992o0Jan9cV

# Starter Code for Catapult Dataset
**Please make a copy of this colab notebook for yourself and put the copy in your own google drive folder(don't leave the copy in the shared google drive!) before you start hacking! This file is shared among all the participants, so don't pollute this file!**

You must run the following two blocks of code. They are used for importing data from the csv file in our google drive.
"""

!pip install -U -q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from google.colab import files

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# catapult.csv https://drive.google.com/open?id=1WE5NAT2_EIRwnVK04ViUSgcc9MPBTebH
data_download_user = drive.CreateFile({'id': '1WE5NAT2_EIRwnVK04ViUSgcc9MPBTebH'})
data_download_user.GetContentFile('catapult.csv')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

"""Read in the data"""

df_original = pd.read_csv('catapult.csv')

"""Take a look at the dataset."""

print(df_original.shape)
df_original.info()

df_original.head()

"""## Data Cleaning"""

#filter out water breaks
df_cleaned = df_original[(df_original["Activity Name"] != "Water Break")]

#filter out invalid distances
df_cleaned = df_cleaned[(df_cleaned["Total Distance"] < 20000) & (df_cleaned["Total Distance"] > 50)]

#filter out bad player loads
df_cleaned = df_cleaned[(df_cleaned["Total Player Load"] < 1250) & (df_cleaned["Total Player Load"] > 50)]

#filter out bad 'Total IMAs'
df_cleaned = df_cleaned[(df_cleaned["Tot IMA"] < 650) & (df_cleaned["Tot IMA"] > 20)]

#filter out bad high speed distance
df_cleaned = df_cleaned[(df_cleaned["High Speed Distance"] < 1000)]

#filter out bad max velocities
df_cleaned = df_cleaned[(df_cleaned["Maximum Velocity"] > 2) & (df_cleaned["Maximum Velocity"] < 8)]

#create spreadsheet of games exclusively
all_games = df_cleaned[(df_cleaned["Period Name"]).str.contains("first half", case=False) |
                       (df_cleaned["Period Name"]).str.contains("second half", case=False) |
                       (df_cleaned["Period Name"]).str.contains("overtime", case=False) |
                       (df_cleaned["Period Name"]).str.match("OT", case=False)
                      ]

#Remove misleading practices
all_games = all_games[(all_games["Period Name"]).str.contains("OT practice", case=False) == False]
all_games = all_games[(all_games["Period Name"]).str.contains("11 v 11", case=False) == False]
all_games = all_games[(all_games["Period Name"]).str.contains("11v11", case=False) == False]
all_games = all_games[(all_games["Period Name"]).str.contains("9 v 9", case=False) == False] 
all_games = all_games[(all_games["Period Name"]).str.contains("9v9", case=False) == False]

"""## Data Visualization"""

#Demonstrate the correlation between Total Distance and Total Player Load
df_cleaned.plot(x='Total Distance', y='Total Player Load', kind ="scatter")
plt.title("Total Distance vs Total Player Load")
plt.show()

#create histogram of maximum velocities
df_cleaned["Maximum Velocity"].plot('hist')
plt.title("Distribution of Maximum Velcocities")
plt.xlabel("Maximum Velocity")
plt.show()
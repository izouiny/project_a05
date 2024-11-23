from ift6758.features import load_advanced_train_test_dataframes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Data download to ease the development of the advanced models

# Part for basic xgboost
"""df2016 = load_events_dataframe(2016)
df2017 = load_events_dataframe(2017)
df2018 = load_events_dataframe(2018)
df2019 = load_events_dataframe(2019)
df2020 = load_events_dataframe(2020)

dtrain = pd.concat([df2016, df2017, df2018], ignore_index=True)
dvalid = df2019
dtest = df2020

#Save files as csv
dtrain.to_csv("data_for_models/training_set2.csv")
dvalid.to_csv("data_for_models/validation_set2.csv")"""

# Part for best xgboost
"""dtrain, dtest = load_advanced_train_test_dataframes()
print("loading done")
#dtrain, dvalid = train_test_split(dtrain, test_size=0.2, random_state=1)

dtest_saison = dtest[dtest["game_type"] == 2]
dtest_serie = dtest[dtest["game_type"] == 3]


dtrain.to_csv("data_for_models/best_xgboost/train.csv")
dvalid.to_csv("data_for_models/best_xgboost/valid.csv")
dtest_saison.to_csv("data_for_models/test_saison.csv")
dtest_serie.to_csv("data_for_models/test_serie.csv")"""
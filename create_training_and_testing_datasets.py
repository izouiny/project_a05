from ift6758.data import load_events_dataframe
from sklearn.model_selection import train_test_split
import pandas as pd

#Load data from seasons 2016 and 2019 for training and validation datasets
data = pd.concat([load_events_dataframe(2016),load_events_dataframe(2017),load_events_dataframe(2018),load_events_dataframe(2019)], ignore_index=True)
#Randomly split the dataset, keeping 80% for training and 20% for validation
train_set, validation_set = train_test_split(data, test_size=0.2, random_state=1)

#Load data from season 2020 for testing dataset
test_set = load_events_dataframe(2020)

#Save files as csv
train_set.to_csv("data_for_models/training_set.csv")
validation_set.to_csv("data_for_models/validation_set.csv")
test_set.to_csv("data_for_models/testing_set.csv")


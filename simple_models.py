#Load libraries and custom code to load events
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import wandb
import random

#Load training set csv
training_set = pd.read_csv('data_for_models/training_set.csv')

#Load validation set csv
validation_set = pd.read_csv('data_for_models/validation_set.csv')

#Model for goal_distance
x_train = training_set[['goal_distance']]
x_val = validation_set[['goal_distance']]
y_train = training_set['is_goal']
y_val = validation_set['is_goal']

#Train the logistic regression model
clf = LogisticRegression()
clf.fit(x_train, y_train)

#Make predictions on the validation set
y_pred = clf.predict(x_val)

#Evaluate model accuracy
print("Accuracy for goal_distance model is:",accuracy_score(y_val, y_pred))
#report = classification_report(y_val, y_pred)



#Model for goal_angle
x_train = training_set[['goal_angle']]
x_val = validation_set[['goal_angle']]
y_train = training_set['is_goal']
y_val = validation_set['is_goal']

#Train the logistic regression model
clf = LogisticRegression()
clf.fit(x_train, y_train)

#Make predictions on the validation set
y_pred = clf.predict(x_val)

#Evaluate model accuracy
print("Accuracy for goal_angle model is:",accuracy_score(y_val, y_pred))

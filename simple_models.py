from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import wandb
import joblib
import matplotlib.pyplot as plt
import os
from ift6758.visualizations import four_graphs_multiple_models
from ift6758.data import load_events_dataframe

#Load data from seasons 2016 and 2019 for training and validation datasets
data = pd.concat([load_events_dataframe(2016),load_events_dataframe(2017),load_events_dataframe(2018),load_events_dataframe(2019)], ignore_index=True)

#Randomly split the training dataset, keeping 80% for training and 20% for validation
training_set, validation_set = train_test_split(data, test_size=0.2, random_state=1)

def simple_logistic_regression(features, save_wandb = False):

    # Prepare the data
    x_train = training_set[features] 
    x_val = validation_set[features] 
    y_train = training_set['is_goal']
    y_val = validation_set['is_goal']

    # Train the logistic regression model
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Predict labels for validation set
    y_pred = clf.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy is: {accuracy}")

    #Predicted probabilities for each type of label
    y_pred_proba = clf.predict_proba(x_val)

    model_name = '_'.join(features)

    #Combine predicted class labels and probabilities into a dataframe for easier interpretation
    data = (y_pred_proba[:, 1],y_val)

    #Define dir where to save models
    model_dir = "simple_models"

    #Save model locally
    joblib.dump(clf, os.path.join(model_dir, '_'.join(features) + ".joblib"))

    if(save_wandb):

        #Initialize a wandb run
        wandb.init(project="ift6758-milestone-2",name=('_'.join(features)+"_logistic_regression"), reinit=True)

        #Log model as an artifact
        artifact = wandb.Artifact('_'.join(features) + '_logistic_regression', type='model')    
        artifact.add_file(os.path.join(model_dir, '_'.join(features) + ".joblib"))
        wandb.log_artifact(artifact)
        wandb.log({"accuracy": accuracy})

        #Finish the run
        wandb.finish()

    return(data)


goal_distance = simple_logistic_regression(['goal_distance'])
goal_angle = simple_logistic_regression(['goal_angle'])
goal_distance_and_angle = simple_logistic_regression(['goal_distance','goal_angle'])

simple_models = {}
simple_models["goal_distance"] = goal_distance
simple_models["goal_angle"] = goal_angle
simple_models["goal_distance_and_angle"] = goal_distance_and_angle

#Creates four graphs of performance for all logistic regression models
four_graphs_multiple_models(simple_models, folder_name="simple_models")
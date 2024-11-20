from ift6758.data import fetch_all_seasons_games_data
from ift6758.visualizations import four_graphs
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.calibration import CalibrationDisplay
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import wandb
import random

# Main advanced models execution file

# Load data (choose which train-valid separation method to use)
def load_data(method="80-20"):
	assert method == "80-20" or method == "year"
	if method == "80-20":
		dtrain = pd.read_csv("./data_for_models/training_set.csv")
		dvalid = pd.read_csv("./data_for_models/validation_set.csv")
	elif method == "year":
		dtrain = pd.read_csv("./data_for_models/training_set2.csv")
		dvalid = pd.read_csv("./data_for_models/validation_set2.csv")
	return dtrain, dvalid


# Question 1 : XGBoost with basic data
def basic_xgboost(dtrain, dvalid, use_wandb=False):
	Xtrain = dtrain[["goal_distance", "goal_angle"]]
	Xvalid = dvalid[["goal_distance", "goal_angle"]]
	Ytrain = dtrain[["is_goal"]]
	Yvalid = dvalid[["is_goal"]]

	# Initialize a wandb run
	if use_wandb:
		wandb.init(
			project="ift6758-milestone-2",
			name=("basic_XGBoost"),
			config={
				"architecture": "XGBoost",
				"dataset": "goal_distance and goal_angle",
				"num_round": 20,
			})

	train = xgb.DMatrix(Xtrain, label=Ytrain, missing=np.nan)
	valid = xgb.DMatrix(Xvalid, label=Yvalid, missing=np.nan)

	evallist = [(train, 'train'), (valid, 'eval')]
	num_round = 20
	bst = xgb.train({}, train, num_round, evallist, early_stopping_rounds=3)
	bst.save_model('./models/0001.model')
	Yproba = bst.predict(valid, iteration_range=(0, bst.best_iteration + 1))
	Ypred = [proba > 0.5 for proba in Yproba]
	accuracy = accuracy_score(Yvalid, Ypred)
	print(accuracy)

	Yvalid = Yvalid.to_numpy().flatten()

	# Log model as an artifact
	if use_wandb:
		artifact = wandb.Artifact('basic_XGBoost', type='model')
		artifact.add_file("./models/0001.model")
		wandb.log_artifact(artifact)
		wandb.log({"accuracy": accuracy})

		four_graphs(Yproba, Yvalid, "basic_xgboost", save_wandb=True)

		# Finish the run
		wandb.finish()
	else:
		four_graphs(Yproba, Yvalid, "basic_xgboost")

	
	return

dtrain, dvalid = load_data("80-20")
basic_xgboost(dtrain, dvalid, use_wandb=True)
from ift6758.data import fetch_all_seasons_games_data
from ift6758.visualizations import four_graphs
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.calibration import CalibrationDisplay
from matplotlib import pyplot as plt
from collections import Counter
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

# Load all data (choose which train-valid separation method to use)
def load_all_data():
	dtrain = pd.read_csv("./data_for_models/best_xgboost/train.csv")
	dvalid = pd.read_csv("./data_for_models/best_xgboost/valid.csv")
	return dtrain, dvalid


def drop_labels(df, labels):
	for label in labels:
		df = df.drop(label, axis=1)
	return df


def one_hot(index, size):
	ar = np.array([0]*size)
	ar[index] = 1
	return ar

def proba_format(Yproba):
	for i in range(len(Yproba)):
		if Yproba[i] < 0:
			Yproba[i] = 0
		elif Yproba[i] > 1:
			Yproba[i] = 1
	return Yproba


def pre_treatment_adv_df(df):
	df = df.iloc[:, 1:]
	df = drop_labels(df, ['game_id', 'season', 'game_type', 'game_date', 'venue',
                          'venue_location', 'away_team_abbrev', 'away_team_name', 'home_team_abbrev', 'home_team_name',
						  'event_id', 'event_idx', 'sort_order', 'max_regulation_periods', 'situation_code'	,'type_code',
						  'type_desc_key', 'zone_code', 'description', 'details_type_code', 'scoring_player_total',
                          'assist1_player_total', 'assist2_player_total', 'goal_side', 'goal_x_coord', 'shooting_player_name',
						  'event_owner_team_id', 'shooting_player_position_code', 'goalie_in_net_name',
						  'goalie_in_net_position_code', 'scoring_player_id', 'scoring_player_name', 'scoring_player_team_id',
                          'scoring_player_position_code', 'assist1_player_id', 'assist1_player_name', 'assist1_player_team_id',
                          'assist1_player_position_code', 'assist2_player_id', 'assist2_player_name', 'assist2_player_team_id',
                          'assist2_player_position_code', 'time_in_period', 'time_remaining',
						  'shooting_player_id', 'shooting_player_team_id', 'goalie_in_net_id', 'goalie_in_net_team_id',
						  'away_team_id', 'home_team_id', 'away_score', 'home_score', 'away_sog', 'home_sog'])

	# Remove str values for xgboost
	mapping = {'REG': 1, 'SO': 3, 'OT': 2}
	df['period_type'] = df['period_type'].map(mapping)
	mapping = {'tip-in': 0, 'wrap-around': 1, 'wrist': 4, 'slap': 6, 'snap': 5, 'backhand': 3, 'deflected': 2}
	df['shot_type'] = df['shot_type'].map(mapping)
	mapping = {'blocked-shot': 1, 'missed-shot': 2, 'giveaway': 3, 'shot-on-goal': 0, 'faceoff': 5, 'hit': 6,
               'takeaway': 4, 'stoppage': 9, 'goal': 10, 'penalty': 7, 'delayed-penalty': 8, 'period-start': 11,
               'period-end': 12}
	df['last_event_type'] = df['last_event_type'].map(mapping)
	return df

"""", ', 'period_number', 'period_type',
       , ,
      , 'is_empty_net', 'is_goal', 
        , ,
       'x_coord', 'y_coord',  'shot_type', 
         'goal_distance',
       'goal_angle', 
       'time_in_period_seconds',
       'game_seconds', , 'last_x', 'last_y',
       'time_since_last_event', 'distance_from_last_event', 'is_rebound',
       'speed', 'last_angle', 'absolute_angle_change',
       'power_play_time_elapsed', 'shooting_team_skaters',
       'opposing_team_skaters'"""

def show_proba(Yproba):
	counts = Counter(Yproba)
	values = list(counts.keys())
	frequencies = list(counts.values())

	plt.bar(values, frequencies)
	plt.xlabel('Values')
	plt.ylabel('Frequency')
	plt.title('Frequency of Values in Vector')
	plt.savefig("./figures/Yproba.png")

class Run:
    def __init__(self, booster, eta, gamma, max_depth):
        self.booster = booster
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.acc = -1
        self.auc = -1

    def __str__(self):
        return f"{self.booster}_{self.eta}_{self.gamma}_{self.max_depth}"

    def set_acc(self, acc):
        self.acc = acc
        return

    def set_auc(self, auc):
        self.auc = auc
        return


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

#dtrain, dvalid = load_data("80-20")
#basic_xgboost(dtrain, dvalid, use_wandb=True)

# Question 2 : XGBoost with all data
def best_xgboost(use_wandb=False):
	dtrain, dval = load_all_data()

	dtrain = pre_treatment_adv_df(dtrain)
	"""dtrain_goals = dtrain[dtrain["is_goal"]==True]
	dtrain_goals_rebound = dtrain_goals[dtrain_goals["is_rebound"]==True]
	dtrain_no_goals = dtrain[dtrain["is_goal"]==False]
	dtrain_no_goals_rebound = dtrain_no_goals[dtrain_no_goals["is_rebound"] == True]
	f = open("./dtrain_goals.txt", "w")
	f.write(str(dtrain_no_goals_rebound.iloc[9]))
	f.write("\n")
	f.write(str(dtrain_goals_rebound.iloc[18]))
	f.close()"""

	dval = pre_treatment_adv_df(dval)

	dtrainx = dtrain.drop("is_goal", axis=1)
	Xtrain = dtrainx.to_numpy()
	Ytrain = dtrain[["is_goal"]].to_numpy()
	dvalx = dval.drop("is_goal", axis=1)
	Xval = dvalx.to_numpy()
	Yval = dval[["is_goal"]].to_numpy()
	print(f"Nombre de buts dans Yval : {np.sum(Yval)}")

	# Hyperparameters to optimize
	booster = ["gbtree", "dart"]
	eta = [0.1, 0.3, 0.5, 0.8]
	gamma = [0, 10, 100]
	max_depth = [4, 6, 8]

	runs = []
	for b in booster:
		for e in eta:
			for g in gamma:
				for md in max_depth:
					run = Run(b, e, g, md)
					runs.append(run)

	#runs = runs[:1]

	perf = ""
	for run in runs:
		model = "best_xgboost_" + str(run)
		perf += "----------------------------------------------------------------------------------------------------\n"
		perf += model + "\n"

		# Initialize a wandb run
		if use_wandb:
			wandb.init(
				project="best xgboost",
				name=(model),
				config={
					"architecture": "XGBoost",
					"dataset": "advanced data",
					"num_round": 10,
					"booster": run.booster,
					"eta": run.eta,
					"gamma": run.gamma,
					"max_depth": run.max_depth})

		train = xgb.DMatrix(Xtrain, label=Ytrain)
		val = xgb.DMatrix(Xval, label=Yval)

		param = {'booster': run.booster, 'eta': run.eta, 'gamma': run.gamma, 'max_depth': run.max_depth}
		evallist = [(train, 'train'), (val, 'eval')]
		#evallist = [(train, 'train')]
		num_round = 10
		bst = xgb.train(param, train, num_round, evallist, early_stopping_rounds=3)
		bst.save_model(f'./models/{model}.model')
		Yproba = bst.predict(val, iteration_range=(0, bst.best_iteration + 1))
		Yproba = proba_format(Yproba)
		show_proba(Yproba)
		Ypred = [proba > 0.5 for proba in Yproba]
		print(f"Nombre de buts dans Ypred : {np.sum(Ypred)}")
		accuracy = accuracy_score(Yval, Ypred)
		print(f'Accuracy for {model} : {str(accuracy)}')
		perf += f"Accuracy : {str(accuracy)}\n"
		run.set_acc(accuracy)
		Yval = Yval.flatten()

		# Log model as an artifact
		if use_wandb:
			artifact = wandb.Artifact(model, type='model')
			artifact.add_file(f"./models/{model}.model")
			wandb.log_artifact(artifact)
			wandb.log({"accuracy": accuracy})

			auc = four_graphs(Yproba, Yval, model, save_wandb=True)

			# Finish the run
			wandb.finish()
		else:
			auc = four_graphs(Yproba, Yval, model)
		perf += f"AUC : {str(auc)}\n"

	f = open("./best_xgboost_perf.txt", "w")
	f.write(perf)
	f.close()
	return

best_xgboost()

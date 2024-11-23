from ift6758.data import fetch_all_seasons_games_data
from ift6758.visualizations import four_graphs, four_graphs_multiple_models
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.calibration import CalibrationDisplay
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
import xgboost as xgb
import wandb
import joblib
import random

# Main advanced models execution file


# Load data (choose which train-valid separation method to use) only used for basic_xgboost
def load_data(method="80-20"):
	assert method == "80-20" or method == "year"
	if method == "80-20":
		dtrain = pd.read_csv("./data_for_models/training_set.csv")
		dvalid = pd.read_csv("./data_for_models/validation_set.csv")
	elif method == "year":
		dtrain = pd.read_csv("./data_for_models/training_set2.csv")
		dvalid = pd.read_csv("./data_for_models/validation_set2.csv")
	return dtrain, dvalid

# Load all data
def load_all_data():
	dtrain = pd.read_csv("./data_for_models/best_xgboost/train.csv")
	dvalid = pd.read_csv("./data_for_models/best_xgboost/valid.csv")
	return dtrain, dvalid

def load_test_sets():
	dtest_saison = pd.read_csv("data_for_models/test_saison.csv")
	dtest_serie = pd.read_csv("data_for_models/test_serie.csv")
	return dtest_saison, dtest_serie

def drop_labels(df, labels):
	for label in labels:
		df = df.drop(label, axis=1)
	return df


def one_hot(index, size):
	ar = np.array([0]*size)
	ar[index] = 1
	return ar

def proba_format(Yproba):
	'''
	Put the predicted quantities by the model between 0 and 1 to interpret them as probabilities
	'''
	for i in range(len(Yproba)):
		if Yproba[i] < 0:
			Yproba[i] = 0
		elif Yproba[i] > 1:
			Yproba[i] = 1
	return Yproba


def make_graphs(b, e, g, m, nr):
	'''
	Make graphs for XGBoost's hyperparameters comparison independently
	'''
	dicts = [b, e, g, m, nr]
	hps = ["booster", "eta", "gamma", "max_depth", "num_rounds"]
	for i in range(5):
		dict = dicts[i]
		hp = hps[i]
		mean_acc = []
		mean_auc = []
		x = []
		for (key, value) in enumerate(dict.items()):
			key = value[0]
			if len(value[1][0]) == 0:
				continue
			else:
				x.append(key)
				accs = np.array(value[1][0])
				mean_acc.append(np.mean(accs))
				aucs = np.array(value[1][1])
				mean_auc.append(np.mean(aucs))

		if len(x) == 0:
			continue

		# Used chat gpt for the generation of this graph
		bar_width = 0.4
		x_positions = np.arange(len(x))

		# Plot bars
		plt.clf()
		bars_acc = plt.bar(x_positions - bar_width / 2, mean_acc, width=bar_width, label='Accuracy', color='blue', alpha=0.7)
		bars_auc = plt.bar(x_positions + bar_width / 2, mean_auc, width=bar_width, label='AUC', color='orange', alpha=0.7)

		# Add values on top of bars
		for bar in bars_acc:
			plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
					 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, color='blue')

		for bar in bars_auc:
			plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
					 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, color='orange')

		# Add labels, title, and legend
		plt.xlabel(f'{hp} configurations')
		plt.ylabel('Metrics')
		plt.title(f'Accuracy and AUC for different {hp} configurations')
		plt.xticks(ticks=x_positions, labels=x)
		plt.ylim(0, 1)
		plt.legend()

		# Show the plot
		plt.tight_layout()
		plt.savefig(f"./figures/best_xgboost/{hp}.png")


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


def show_proba(Yproba):
	'''
	Visualize your model's predicted probabilities
	Yproba
	'''
	counts = Counter(Yproba)
	values = list(counts.keys())
	frequencies = list(counts.values())

	plt.bar(values, frequencies)
	plt.xlabel('Values')
	plt.ylabel('Frequency')
	plt.title('Frequency of Values in Vector')
	plt.savefig("./figures/Yproba.png")

class Run:
	'''
	Run object to keep in memory every hp when optimizing
	'''
    def __init__(self, booster, eta, gamma, max_depth, num_rounds):
        self.booster = booster
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.num_rounds = num_rounds
        self.acc = -1
        self.auc = -1

    def __str__(self):
        return f"{self.booster}_{self.eta}_{self.gamma}_{self.max_depth}_{self.num_rounds}"

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
#basic_xgboost(dtrain, dvalid, use_wandb=False)

# Question 2 : XGBoost with all data
def best_xgboost(use_wandb=False):
	dtrain, dval = load_all_data()

	dtrain = pre_treatment_adv_df(dtrain)
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
	max_depth = [4, 6, 8, 12]
	num_rounds = [2, 5, 10, 15, 20, 30]
	booster = ["gbtree"]
	eta = [0.3]
	gamma = [0]
	max_depth = [8]

	# Each hyperparameter value has its own list of accuracies and auc. Ex: ([accuracy], [auc])
	booster_v = {"gbtree": ([], []), "dart": ([], [])}
	eta_v = {0.1: ([], []), 0.3: ([], []), 0.5: ([], []), 0.8: ([], [])}
	gamma_v = {0: ([], []), 10: ([], []), 100: ([], [])}
	max_depth_v = {4: ([], []), 6: ([], []), 8: ([], []), 12: ([], [])}
	num_rounds_v = {2: ([], []), 5: ([], []), 10: ([], []), 15: ([], []), 20: ([], []), 30: ([], [])}

	# Create Run objects to keep track of hyperparameters selection
	runs = []
	for b in booster:
		for e in eta:
			for g in gamma:
				for md in max_depth:
					for nr in num_rounds:
						run = Run(b, e, g, md, nr)
						runs.append(run)

	perf = ""
	models = {}
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
					"num_round": run.num_rounds,
					"booster": run.booster,
					"eta": run.eta,
					"gamma": run.gamma,
					"max_depth": run.max_depth})

		train = xgb.DMatrix(Xtrain, label=Ytrain)
		val = xgb.DMatrix(Xval, label=Yval)

		param = {'booster': run.booster, 'eta': run.eta, 'gamma': run.gamma, 'max_depth': run.max_depth}
		evallist = [(train, 'train'), (val, 'eval')]
		num_round = run.num_rounds
		bst = xgb.train(param, train, num_round, evallist, early_stopping_rounds=3)
		bst.save_model(f'./models/{model}.model')
		Yproba = bst.predict(val, iteration_range=(0, bst.best_iteration + 1))

		Yproba = proba_format(Yproba)
		#show_proba(Yproba)
		Ypred = [proba > 0.5 for proba in Yproba]
		print(f"Nombre de buts dans Ypred : {np.sum(Ypred)}")
		accuracy = accuracy_score(Yval, Ypred)
		print(f'Accuracy for {model} : {str(accuracy)}')
		perf += f"Accuracy : {str(accuracy)}\n"
		run.set_acc(accuracy)
		Yval = Yval.flatten()
		auc = 0
		models[model] = (Yproba, Yval)

		# Log model as an artifact
		if use_wandb:
			artifact = wandb.Artifact(model, type='model')
			artifact.add_file(f"./models/{model}.model")
			wandb.log_artifact(artifact)
			wandb.log({"accuracy": accuracy})

			auc = four_graphs(Yproba, Yval, model, save_wandb=True)
			wandb.log({"auc": auc})
			
			# Finish the run
			wandb.finish()
		else:
			auc = four_graphs(Yproba, Yval, model)
			booster_v[run.booster][0].append(float(accuracy))
			booster_v[run.booster][1].append(float(auc))
			eta_v[run.eta][0].append(float(accuracy))
			eta_v[run.eta][1].append(float(auc))
			gamma_v[run.gamma][0].append(float(accuracy))
			gamma_v[run.gamma][1].append(float(auc))
			max_depth_v[run.max_depth][0].append(float(accuracy))
			max_depth_v[run.max_depth][1].append(float(auc))
			num_rounds_v[run.num_rounds][0].append(float(accuracy))
			num_rounds_v[run.num_rounds][1].append(float(auc))

		perf += f"AUC : {str(auc)}\n"


	f = open("./best_xgboost_perf.txt", "w")
	f.write(perf)
	f.close()

	if not use_wandb:
		make_graphs(booster_v, eta_v, gamma_v, max_depth_v, num_rounds_v)
		four_graphs_multiple_models_hp(models, "best_xgboost_gbtree_0.3_0_8", "best_xgboost")
	return

#best_xgboost()

def feature_selection_graph():
	selection_method = ["no_selection", "var_0.160", "var_0.227", "k_best_5", "k_best_10", "k_best_15", "k_best_20"]
	auc = [0.7746601291087247, 0.7746351466788768, 0.7736448506259643, 0.7498530247824737, 0.7662138333133821,
		   0.7686317656553328, 0.7740964680709775]

	# Used chat gpt for the generation of this graph

	# Bar graph
	plt.figure(figsize=(10, 6))
	bars = plt.bar(selection_method, auc, color='skyblue', edgecolor='black')

	# Annotate AUC values on top of each bar
	for bar, value in zip(bars, auc):
		plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.4f}',
				 ha='center', va='bottom', fontsize=10)

	# Customize y-axis scale
	y_min = min(auc) - .01
	y_max = max(auc) + .01
	plt.ylim(y_min, y_max)

	# Add labels and title
	plt.xlabel('Selection Method', fontsize=12)
	plt.ylabel('AUC', fontsize=12)
	plt.title('XGBoost AUC by feature selection method', fontsize=14)
	plt.xticks(rotation=45, fontsize=10)

	# Show plot
	plt.tight_layout()
	plt.savefig("./figures/feature_selection")
	return

#feature_selection_graph()

# Different feature selection strategies
def identity(data):
    return data, "identity"

def variance_thresh(data, t=.8 * (1 - .8)):
    sel = VarianceThreshold(threshold=(t))
    new_data = sel.fit_transform(data)
    selected_features = data.columns[sel.get_support()]
    return data[selected_features], f"variance_t_{str(round(t, 3))}"

def select_k(data, label, k=10):
    sel = SelectKBest(k=k)
    new_data = sel.fit_transform(data, label)
    selected_features = data.columns[sel.get_support()]
    return data[selected_features], f"select_k_{str(k)}", selected_features


# Question 3 feature selection
def xgboost_feat_sel(selection_fct, use_wandb=False):
	dtrain, dval = load_all_data()

	dtrain = pre_treatment_adv_df(dtrain)
	dtrain = dtrain.apply(lambda col: col.fillna(col.mean()), axis=0)
	dval = pre_treatment_adv_df(dval)
	dval = dval.apply(lambda col: col.fillna(col.mean()), axis=0)



	dtrainx = dtrain.drop("is_goal", axis=1)
	dtrainx, selection_name = selection_fct(dtrainx)
	dvalx = dval.drop("is_goal", axis=1)
	dvalx, selection_name = selection_fct(dvalx)

	# Print the columns names
	print("Columns names:", dtrainx.columns)

	# Write out the selected features
	f = open(f"./sel_feat_{selection_name}.txt", "w")
	f.write(str(dtrainx.columns))
	f.close()

	Xtrain = dtrainx.to_numpy()
	Xval = dvalx.to_numpy()
	Ytrain = dtrain[["is_goal"]].to_numpy()
	Yval = dval[["is_goal"]].to_numpy()

	print(f"Nombre de buts dans Yval : {np.sum(Yval)}")

	model = f"xgboost_{selection_name}"

	# Initialize a wandb run
	if use_wandb:
		wandb.init(
			project="xgboost_feature_selection",
			name=(model),
			config={
				"architecture": "XGBoost",
				"dataset": "advanced data selected",
				'objective': 'binary:logistic',
				"selection": selection_name,
				"num_round": 10,
				"booster": 'gbtree',
				"eta": 0.3,
				"gamma": 0,
				"max_depth": 8})

	train = xgb.DMatrix(Xtrain, label=Ytrain)
	val = xgb.DMatrix(Xval, label=Yval)

	param = {'booster': 'gbtree', 'eta': 0.3, 'gamma': 0, 'max_depth': 8}
	evallist = [(train, 'train'), (val, 'eval')]
	num_round = 10

	bst = xgb.train(param, train, num_round, evallist, early_stopping_rounds=3)
	bst.save_model(f'./models/{model}.model')
	Yproba = bst.predict(val, iteration_range=(0, bst.best_iteration + 1))

	Yproba = proba_format(Yproba)
	# show_proba(Yproba)
	Ypred = [proba > 0.5 for proba in Yproba]
	print(f"Nombre de buts dans Ypred : {np.sum(Ypred)}")
	accuracy = accuracy_score(Yval, Ypred)
	print(f'Accuracy for {model} : {str(accuracy)}')
	Yval = Yval.flatten()
	auc = 0

	# Log model as an artifact
	if use_wandb:
		joblib.dump(bst, f"models/{model}.pkl")
		artifact = wandb.Artifact(model, type="model")
		artifact.add_file(f"models/{model}.pkl")
		#artifact = wandb.Artifact(model, type='model')
		#artifact.add_file(f"./models/{model}.model")
		wandb.log_artifact(artifact)
		wandb.log({"accuracy": accuracy})

		auc = four_graphs(Yproba, Yval, model, save_wandb=True)
		wandb.log({"auc": auc})


		# Finish the run
		wandb.finish()
	else:
		auc = four_graphs(Yproba, Yval, model)
		print(f"AUC for {model} : {str(auc)}")

	return

#xgboost_feat_sel(identity, use_wandb=False)


# Test our best XGBoost model and save the returned probabilities
def test_best_model():
	dtrain, dval = load_all_data()
	dtest_saison, dtest_serie = load_test_sets()

	dtrain = pre_treatment_adv_df(dtrain)
	dtest_saison = pre_treatment_adv_df(dtest_saison)
	dtest_serie = pre_treatment_adv_df(dtest_serie)


	X_train = dtrain.drop("is_goal", axis=1)
	X_test_saison = dtest_saison.drop("is_goal", axis=1)
	X_test_serie = dtest_serie.drop("is_goal", axis=1)

	X_train = X_train.to_numpy()
	X_test_saison = X_test_saison.to_numpy()
	X_test_serie = X_test_serie.to_numpy()
	Y_train = dtrain[["is_goal"]].to_numpy()
	Y_test_saison = dtest_saison[["is_goal"]].to_numpy()
	Y_test_serie = dtest_serie[["is_goal"]].to_numpy()

	print(f"Nombre de buts dans test_saison : {np.sum(Y_test_saison)}")
	print(f"Nombre de buts dans test_serie : {np.sum(Y_test_serie)}")

	model = f"xgboost_final_test"

	train = xgb.DMatrix(X_train, label=Y_train)
	test_saison = xgb.DMatrix(X_test_saison)
	test_serie = xgb.DMatrix(X_test_serie)

	param = {'booster': 'gbtree', 'eta': 0.3, 'gamma': 0, 'max_depth': 8}
	#evallist = [(train, 'train'), (, 'eval')]
	num_round = 10

	#bst = xgb.train(param, train, num_round, evallist, early_stopping_rounds=3)
	bst = xgb.train(param, train, num_round)
	Y_proba_saison = bst.predict(test_saison)
	Y_proba_serie = bst.predict(test_serie)

	Y_proba_saison = proba_format(Y_proba_saison)
	Y_proba_serie = proba_format(Y_proba_serie)


	np.save('Y_proba_saison.npy', Y_proba_saison)
	np.save('Y_proba_serie.npy', Y_proba_serie)

	Y_pred_saison = [proba > 0.5 for proba in Y_proba_saison]
	Y_pred_serie = [proba > 0.5 for proba in Y_proba_serie]
	print(f"Nombre de buts prédits dans saison : {np.sum(Y_pred_saison)}")
	print(f"Nombre de buts prédits dans serie : {np.sum(Y_pred_serie)}")
	accuracy_saison = accuracy_score(Y_test_saison, Y_pred_saison)
	accuracy_serie = accuracy_score(Y_test_serie, Y_pred_serie)
	print(f'Accuracy for saison : {str(accuracy_saison)}')
	print(f'Accuracy for serie : {str(accuracy_serie)}')
	Y_test_saison = Y_test_saison.flatten()
	Y_test_serie =Y_test_serie.flatten()

	np.save('Y_label_saison.npy', Y_test_saison)
	np.save('Y_label_serie.npy', Y_test_serie)

	auc_saison = four_graphs(Y_proba_saison, Y_test_saison, "test_saison")
	auc_serie = four_graphs(Y_proba_serie, Y_test_serie, "test_serie")
	print(f"AUC for saison : {str(auc_saison)}")
	print(f"AUC for serie : {str(auc_serie)}")

	return

#test_best_model()
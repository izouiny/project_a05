from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.calibration import CalibrationDisplay
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import wandb
import random
import os

# Visualization of the four graphs asked
def four_graphs(Yproba, Yvalid, model_name, save_wandb=False):
	"""
	Generate the four asked graphs from the probabilities predictions, the true labels, the model name and a boolean
	value which indicates if you want to save the figures to wandb (True) or locally (False).

	Yproba and Yvalid must be numpy arrays
	"""
	Yrandom_pred = [random.randint(0, 1) for i in range(len(Yproba))]  # random classifier prediction

	# Start with the ROC curv ##########################################################################################
	fig_name = "roc_" + model_name + ".png"
	# Compute the false positive rate (fpr), true positive rate (tpr), and thresholds
	fpr, tpr, thresholds = roc_curve(Yvalid, Yproba)
	rand_fpr, rand_tpr, rand_thresholds = roc_curve(Yvalid, Yrandom_pred)

	# Calculate the Area Under the Curve (AUC)
	roc_auc = auc(fpr, tpr)
	rand_roc_auc = auc(rand_fpr, rand_tpr)

	# Plot the ROC curve
	plt.figure(figsize=(12, 9))
	plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

	# Add the baseline for a random classifier
	plt.plot(rand_fpr, rand_tpr, color='red', linestyle='--', label=f'Random Classifier (AUC = {rand_roc_auc:.2f})')

	# Aesthetics and labels
	plt.title(f'Receiver Operating Characteristic (ROC) Curve\nfor {model_name}', fontsize=16, fontweight='bold')
	plt.xlabel('False Positive Rate (FPR)', fontsize=14)
	plt.ylabel('True Positive Rate (TPR)', fontsize=14)
	plt.legend(loc='lower right', fontsize=12)
	plt.grid(alpha=0.3)
	plt.tight_layout()
	if save_wandb:
		wandb.log({"roc_curve": plt})
	else:
		plt.savefig("./figures/" + fig_name)

	# Then, the goal rate as a function of the probability of goal centile #############################################
	fig_name = "goal_rate_vs_proba_" + model_name + ".png"
	data = {'Yproba': Yproba, 'Yvalid': Yvalid}
	df = pd.DataFrame(data)
	# Define centiles
	df['centile'] = pd.qcut(df['Yproba'], 100, labels=False, duplicates="drop")

	# Calculate goal rate for each centile
	centile_goal_rate = df.groupby('centile').apply(
		lambda group: group['Yvalid'].sum() / len(group)
	).reset_index(name='goal_rate')

	# Plot goal rate as a function of centiles
	plt.figure(figsize=(12, 9))
	plt.plot(centile_goal_rate['centile'], centile_goal_rate['goal_rate'], label=f'Goal rate for {model_name}')

	# Add labels and formatting
	plt.title(f'Goal Rate as a Function of Probability Centiles\nfor {model_name}', fontsize=16, fontweight='bold')
	plt.xlabel('Probability Centile', fontsize=14)
	plt.ylabel('Goal Rate', fontsize=14)
	plt.grid(alpha=0.3)
	plt.xticks(ticks=np.arange(0, 101, 10), labels=np.arange(0, 101, 10))
	plt.legend(fontsize=12)
	plt.tight_layout()
	if save_wandb:
		wandb.log({"goal_rate_vs_proba_centile": plt})
	else:
		plt.savefig("./figures/" + fig_name)

	# Then, the cumulative % of goals as a function of the probability of goal centile #################################
	fig_name = "cumul_goal_prop_vs_proba_" + model_name + ".png"
	# Define centiles
	data = {'Yproba': Yproba, 'Yvalid': Yvalid}
	df = pd.DataFrame(data)
	df['centile'] = pd.qcut(df['Yproba'], 100, labels=False, duplicates="drop")

	# Calculate the number of goals per centile
	centile_goals = df[df['Yvalid'] == True].groupby('centile')['Yvalid'].count().reset_index(name='goals')

	# Calculate the cumulative percentage of goals
	centile_goals['cumulative_goals'] = centile_goals['goals'].cumsum()
	centile_goals['cumulative_percent'] = 100 * centile_goals['cumulative_goals'] / centile_goals['goals'].sum()

	# Plot cumulative percentage of goals as a function of centiles
	plt.figure(figsize=(12, 9))
	plt.plot(centile_goals['centile'], centile_goals['cumulative_percent'], label=f'Cumulative % of Goals for {model_name}')

	# Add labels and formatting
	plt.title(f'Cumulative % of Goals as a Function of Probability Centiles\nfor {model_name}', fontsize=16, fontweight='bold')
	plt.xlabel('Probability Centile', fontsize=14)
	plt.ylabel('Cumulative % of Goals', fontsize=14)
	plt.grid(alpha=0.3)
	plt.xticks(ticks=np.arange(0, 101, 10), labels=np.arange(0, 101, 10))
	plt.legend(fontsize=12)
	plt.tight_layout()
	if save_wandb:
		wandb.log({"cumul_goal_proba_vs_proba_centile": plt})
	else:
		plt.savefig("./figures/" + fig_name)

	# Finally, the calibration curve ###################################################################################
	fig_name = "calibration_curve_" + model_name + ".png"
	disp = CalibrationDisplay.from_predictions(Yvalid, Yproba, n_bins=10, name=model_name)

	plt.gcf().set_size_inches(12, 9)
	plt.title(f'Reliability Diagram (Calibration Curve)\nfor {model_name}', fontsize=16)
	plt.xlabel('Mean Predicted Probability', fontsize=14)
	plt.ylabel('Fraction of Positives', fontsize=14)
	plt.grid(alpha=0.3)
	plt.tight_layout()
	if save_wandb:
		# wandb.log({"calibration curve": plt})  # Problem with this part
	    print("Can't save calibration curve to wandb")
	else:
		plt.savefig("./figures/" + fig_name)
	plt.close()
	return roc_auc
	
	
	
def four_graphs_multiple_models(models, folder_name = "models"):
    """
    Generate combined plots for ROC curve, goal rate, cumulative % goals, and calibration curves 
    for multiple models (curves) on the same figure.
    
    Parameters:
        models (dict): A dictionary where keys are model names and values are tuples of (Yproba, Yvalid).
                       Example: {'model1': (Yproba1, Yvalid1), 'model2': (Yproba2, Yvalid2)}
    """
    os.makedirs(f"./figures/{folder_name}", exist_ok=True)  # Ensure output directory exists

    # Plot 1: Combined ROC Curve
    plt.figure(figsize=(12, 9))
    for model_name, (Yproba, Yvalid) in models.items():
        fpr, tpr, _ = roc_curve(Yvalid, Yproba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1, label='Random Classifier')
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./figures/{folder_name}/roc_curve.png")
    plt.close()

    # Plot 2: Combined Goal Rate as a Function of Probability Centiles
    plt.figure(figsize=(12, 9))
    for model_name, (Yproba, Yvalid) in models.items():
        df = pd.DataFrame({'Yproba': Yproba, 'Yvalid': Yvalid})
        df['centile'] = pd.qcut(df['Yproba'], 100, labels=False, duplicates="drop")
        centile_goal_rate = df.groupby('centile').apply(lambda group: group['Yvalid'].mean()).reset_index(name='goal_rate')
        plt.plot(centile_goal_rate['centile'], centile_goal_rate['goal_rate']*100, label=f'Goal Rate for {model_name}')
    random_goal_rate = Yvalid.mean() * 100  # Overall goal rate in percentage
    plt.axhline(y=random_goal_rate, color='grey', linestyle='--', label='Random Classifier')
    plt.title('Goal Rate vs Probability Centiles', fontsize=16, fontweight='bold')
    plt.xlabel('Probability Centile', fontsize=14)
    plt.ylabel('Goal Rate', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./figures/{folder_name}/goal_rate.png")
    plt.close()

    # Plot 3: Combined Cumulative % of Goals as a Function of Probability Centiles
    plt.figure(figsize=(12, 9))
    for model_name, (Yproba, Yvalid) in models.items():
        df = pd.DataFrame({'Yproba': Yproba, 'Yvalid': Yvalid})
        df['centile'] = pd.qcut(df['Yproba'], 100, labels=False, duplicates="drop")
        centile_goals = df[df['Yvalid'] == 1].groupby('centile')['Yvalid'].count().reset_index(name='goals')
        centile_goals['cumulative_goals'] = centile_goals['goals'].cumsum()
        centile_goals['cumulative_percent'] = 100 * centile_goals['cumulative_goals'] / centile_goals['goals'].sum()
        plt.plot(centile_goals['centile'], centile_goals['cumulative_percent'], label=f'Cumulative % Goals for {model_name}')
    plt.plot(range(100), range(1, 101), color='grey', linestyle='--', label='Random Classifier')
    plt.title('Cumulative % Goals vs Probability Centiles', fontsize=16, fontweight='bold')
    plt.xlabel('Probability Centile', fontsize=14)
    plt.ylabel('Cumulative % Goals', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./figures/{folder_name}/cumul_goal_prop.png")
    plt.close()

    # Plot 4: Combined Calibration Curve
    fig, ax = plt.subplots(figsize=(12, 9))
    for model_name, (Yproba, Yvalid) in models.items():
        CalibrationDisplay.from_predictions(Yvalid, Yproba, n_bins=10, name=model_name, ax=ax)
    ax.set_title('Calibration Curve', fontsize=16, fontweight='bold')
    ax.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax.set_ylabel('Fraction of Positives', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./figures/{folder_name}/calibration_curve.png")
    plt.close()


# Function for hyperparameters optimization visualization only
# Save ROC curve with selected_model's curve in red and all other models' in blue
def four_graphs_multiple_models_hp(models, selected_model_name, folder_name="models"):
	"""
	Generate combined plots for ROC curve, goal rate, cumulative % goals, and calibration curves
	for multiple models (curves) on the same figure.

	Parameters:
		models (dict): A dictionary where keys are model names and values are tuples of (Yproba, Yvalid).
					   Example: {'model1': (Yproba1, Yvalid1), 'model2': (Yproba2, Yvalid2)}
	"""
	os.makedirs(f"./figures/{folder_name}", exist_ok=True)  # Ensure output directory exists

	# Plot 1: Combined ROC Curve
	plt.figure(figsize=(12, 9))
	for model_name, (Yproba, Yvalid) in models.items():
		fpr, tpr, _ = roc_curve(Yvalid, Yproba)
		roc_auc = auc(fpr, tpr)
		if model_name == selected_model_name:
			plt.plot(fpr, tpr, lw=2, label=f'Selected model: {model_name} (AUC = {roc_auc:.4f})', color="r", zorder=2)
		else:
			plt.plot(fpr, tpr, lw=2, color="b", zorder=1)

	plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1, label='Random Classifier')
	plt.title('ROC Curve for all different hyperparameter configurations', fontsize=16, fontweight='bold')
	plt.xlabel('False Positive Rate (FPR)', fontsize=14)
	plt.ylabel('True Positive Rate (TPR)', fontsize=14)
	plt.legend(loc='lower right', fontsize=12)
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(f"./figures/{folder_name}/roc_curve.png")
	plt.close()


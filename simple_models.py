from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibrationDisplay
#from sklearn.model_selection import train_test_split
import pandas as pd
import wandb
import joblib
import matplotlib.pyplot as plt
import os

# Load training set CSV
training_set = pd.read_csv('data_for_models/training_set.csv')

# Load validation set CSV
validation_set = pd.read_csv('data_for_models/validation_set.csv')

def simple_logistic_regression(features):

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

    #Combine predicted class labels and probabilities into a dataframe for easier interpretation
    summary_df = pd.DataFrame({
        'real_label':y_val,
        'predicted_label': y_pred,
        'probability_no_goal': y_pred_proba[:, 0],
        'probability_goal': y_pred_proba[:, 1]
    })

    # Save dataframe
    model_dir = "simple_models"
    summary_df.to_csv(os.path.join(model_dir, '_'.join(features) + "_model.csv"), index=False)

    #Create the calibration display from the validation set
    CalibrationDisplay.from_estimator(clf, x_val, y_val, n_bins=10, strategy='uniform')

    #Add plot titles and labels
    plt.title('Reliability Diagram (Calibration Curve)', fontsize=16)
    plt.xlabel('Mean Predicted Probability', fontsize=14)
    plt.ylabel('Fraction of Positives', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, '_'.join(features) + ".png"), dpi=300, bbox_inches='tight')

    #Save model locally
    joblib.dump(clf, os.path.join(model_dir, '_'.join(features) + ".joblib"))

    #Initialize a wandb run
    wandb.init(project="ift6758-milestone-2",name=('_'.join(features)+"_logistic_regression"), reinit=True)

    #Log model as an artifact
    artifact = wandb.Artifact('_'.join(features) + '_logistic_regression', type='model')    
    artifact.add_file(os.path.join(model_dir, '_'.join(features) + ".joblib"))
    wandb.log_artifact(artifact)
    wandb.log({"accuracy": accuracy})

    #Finish the run
    wandb.finish()


simple_logistic_regression(['goal_distance'])
simple_logistic_regression(['goal_angle'])
simple_logistic_regression(['goal_distance','goal_angle'])